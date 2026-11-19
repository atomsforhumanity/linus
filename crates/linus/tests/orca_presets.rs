//! The orca PCM preset and ORCA's solvent table (data/orca/pcm.json, solvent_eps.csv,
//! solvent_aliases.csv) against ORCA 6.0.0's own numbers.

use linus::solvent::SolventSettings;
use linus_solvent::db::{lookup, EpsConvention};
use linus_solvent::PcmPreset;

#[test]
fn orca_solvent_table() {
    let w = lookup("water").unwrap();
    assert_eq!(w.eps_for(EpsConvention::OrcaSmd), Some(78.355));
    assert_eq!(w.eps_for(EpsConvention::OrcaCpcm), Some(80.151));
    let t = lookup("thf").unwrap();
    assert_eq!(t.eps_for(EpsConvention::OrcaCpcm), Some(7.565));
    assert_eq!(t.eps_for(EpsConvention::OrcaSmd), Some(7.4257));
    // ORCA 6.0.0's CPCM refractive index of DMSO (before the 2021 Minnesota correction)
    let d = lookup("dmso").unwrap();
    assert_eq!(d.refrac_orca_cpcm, Some(1.417));
    // names only ORCA resolves
    assert_eq!(lookup("BUTANOL").map(|s| s.key.as_str()), Some("BUTANOL_1"));
    assert_eq!(
        lookup("isooctane").map(|s| s.key.as_str()),
        Some("TRIMETHYLPENTANE_2_2_4")
    );
}

#[test]
fn orca_pcm_preset_grids() {
    // ORCA 6.0.0 picks 194 points for O and 110 for H at its point density (water, C-PCM)
    let so = SolventSettings {
        eps: Some(Some(80.151)),
        preset: Some(PcmPreset::Orca),
        ..Default::default()
    };
    let st = so.resolve(&[8, 1, 1]).unwrap();
    assert_eq!(st.lebedev, vec![194, 110, 110]);
    assert!(st.point_potential);
}
