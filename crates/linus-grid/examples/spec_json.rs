//! Print a preset's GridSpec as JSON (to be edited and fed back via grid_dump json:<file>).
//! usage: spec_json <preset>
fn main() {
    let p = std::env::args().nth(1).unwrap_or("linus:2".into());
    let (name, arg) = p.split_once(':').unwrap_or((&p, ""));
    let spec = match name {
        "pyscf" => linus_grid::GridSpec::pyscf(arg.parse().unwrap_or(3)),
        "psi4" => linus_grid::GridSpec::psi4_default(),
        "orca" => linus_grid::GridSpec::orca_defgrid(arg.parse().unwrap_or(2)),
        "orca-guess" => linus_grid::GridSpec::orca_defgrid_guess(arg.parse().unwrap_or(2)),
        _ => linus_grid::GridSpec::linus(arg.parse().unwrap_or(2)),
    };
    println!("{}", serde_json::to_string_pretty(&spec).unwrap());
}
