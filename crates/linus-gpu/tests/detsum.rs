//! The device accumulator against the CPU's (linus-num's `FixedAcc`): same bits, in any order.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::detsum::DetAcc;
use linus_num::det::FixedAcc;

#[test]
fn scatter_add_matches_cpu_bits_in_any_order() {
    let Some(gpu) = common::gpu() else { return };
    let (n, count) = (1000usize, 400_000usize);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let idx: Vec<i64> = (0..count)
        .map(|_| (rng.next_u64() % n as u64) as i64)
        .collect();
    let v: Vec<f64> = (0..count).map(|_| rng.wide()).collect();

    let mut cpu = FixedAcc::new(n);
    for (&i, &x) in idx.iter().zip(&v) {
        cpu.add1(i as usize, x);
        cpu.tick(1);
    }
    let cpu = cpu.to_f64();

    let mut acc = DetAcc::new(&gpu, n).unwrap();
    acc.scatter_add(&gpu, &gpu.upload(&idx).unwrap(), &gpu.upload(&v).unwrap())
        .unwrap();
    let g1 = acc.to_f64(&gpu).unwrap();

    // reversed order, in two launches
    let (ir, vr): (Vec<i64>, Vec<f64>) = idx
        .iter()
        .rev()
        .zip(v.iter().rev())
        .map(|(&i, &x)| (i, x))
        .unzip();
    let h = count / 3;
    let mut acc2 = DetAcc::new(&gpu, n).unwrap();
    acc2.scatter_add(
        &gpu,
        &gpu.upload(&ir[..h]).unwrap(),
        &gpu.upload(&vr[..h]).unwrap(),
    )
    .unwrap();
    acc2.scatter_add(
        &gpu,
        &gpu.upload(&ir[h..]).unwrap(),
        &gpu.upload(&vr[h..]).unwrap(),
    )
    .unwrap();
    let g2 = acc2.to_f64(&gpu).unwrap();

    for i in 0..n {
        assert_eq!(
            g1[i].to_bits(),
            cpu[i].to_bits(),
            "element {i}: gpu {} cpu {}",
            g1[i],
            cpu[i]
        );
        assert_eq!(g2[i].to_bits(), cpu[i].to_bits(), "element {i}, reversed");
    }
}

#[test]
fn block_sums_match_cpu_bits_at_any_block_size() {
    let Some(gpu) = common::gpu() else { return };
    let count = 1 << 20;
    let mut rng = Rng(12345);
    let v: Vec<f64> = (0..count).map(|_| rng.wide()).collect();
    let mut cpu = FixedAcc::new(1);
    for &x in &v {
        cpu.add1(0, x);
        cpu.tick(1);
    }
    let want = cpu.value(0);
    let dv = gpu.upload(&v).unwrap();
    for block in [32u32, 128, 1024] {
        let mut acc = DetAcc::new(&gpu, 1).unwrap();
        acc.sum_into_first(&gpu, &dv, block).unwrap();
        let got = acc.to_f64(&gpu).unwrap()[0];
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "block {block}: {got} vs {want}"
        );
    }
    // the plain float sum differs in the last bits; the fixed-point sum is exact to 2^-82 per term
    let naive: f64 = v.iter().sum();
    assert!((naive - want).abs() < 1e-9 * want.abs().max(1.0));
}

#[test]
fn one_element_takes_millions_of_additions() {
    // 2^21 additions of values with large low words would overflow a two-word device
    // accumulator (2^63 / 2^52 = 2048 additions); the three-word one stays exact
    let Some(gpu) = common::gpu() else { return };
    let count = 1usize << 21;
    let mut rng = Rng(777);
    let v: Vec<f64> = (0..count)
        .map(|_| {
            0.999_999
                * (2.0f64.powi(-30))
                * (1.0 + rng.uniform())
                * if rng.uniform() < 0.3 { -1.0 } else { 1.0 }
        })
        .collect();
    let mut cpu = FixedAcc::new(1);
    for &x in &v {
        cpu.add1(0, x);
        cpu.tick(1);
    }
    let mut acc = DetAcc::new(&gpu, 1).unwrap();
    acc.scatter_add(
        &gpu,
        &gpu.upload(&vec![0i64; count]).unwrap(),
        &gpu.upload(&v).unwrap(),
    )
    .unwrap();
    assert_eq!(
        acc.to_f64(&gpu).unwrap()[0].to_bits(),
        cpu.value(0).to_bits()
    );
}

#[test]
fn an_invalid_value_poisons_the_accumulator() {
    // the CPU's FixedAcc reads NaN everywhere once a value it cannot hold was added; so must the
    // device's, instead of clamping it into a finite number
    let Some(gpu) = common::gpu() else { return };
    let n = 16usize;
    for bad in [f64::NAN, f64::INFINITY, -f64::INFINITY, 1e10, -5e9] {
        let idx: Vec<i64> = (0..64).map(|i| (i % n) as i64).collect();
        let mut v: Vec<f64> = (0..64).map(|i| 0.25 * i as f64 - 3.0).collect();
        v[37] = bad;
        let mut cpu = FixedAcc::new(n);
        for (&i, &x) in idx.iter().zip(&v) {
            cpu.add1(i as usize, x);
            cpu.tick(1);
        }
        assert!(cpu.to_f64().iter().all(|x| x.is_nan()), "the CPU's, {bad}");
        let mut acc = DetAcc::new(&gpu, n).unwrap();
        acc.scatter_add(&gpu, &gpu.upload(&idx).unwrap(), &gpu.upload(&v).unwrap())
            .unwrap();
        let got = acc.to_f64(&gpu).unwrap();
        assert!(
            got.iter().all(|x| x.is_nan()),
            "scatter_add of {bad}: {got:?}"
        );
        // zeroing clears the flag with the words
        acc.zero(&gpu).unwrap();
        assert!(acc.to_f64(&gpu).unwrap().iter().all(|&x| x == 0.0));

        // the block sum: one thread's value is invalid
        let mut w = vec![1.0f64; 4096];
        w[1234] = bad;
        let mut acc = DetAcc::new(&gpu, 1).unwrap();
        acc.sum_into_first(&gpu, &gpu.upload(&w).unwrap(), 128)
            .unwrap();
        let got = acc.to_f64(&gpu).unwrap()[0];
        assert!(got.is_nan(), "sum_into_first of {bad}: {got}");
    }
    // the largest valid magnitude stays valid
    let mut acc = DetAcc::new(&gpu, 1).unwrap();
    acc.sum_into_first(&gpu, &gpu.upload(&[4294967296.0, -1.0]).unwrap(), 32)
        .unwrap();
    assert_eq!(acc.to_f64(&gpu).unwrap()[0], 4294967295.0);
}
