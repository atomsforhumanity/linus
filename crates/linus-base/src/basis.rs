//! Basis set errors as [`LinusError`]s with actionable messages. The basis set library itself
//! (loading by name from a search path, composites `a|b`, listing, did-you-mean, file names)
//! is [`linus_ints::basis`].

use crate::error::LinusError;
use linus_ints::basis::{available_basis_sets, did_you_mean, BasisError};

/// A basis set error with an actionable message: did-you-mean for unknown names (among the
/// sets on the search path that was searched), element symbols, and the def2 sets (which carry
/// ECPs) for missing elements. Files that cannot be read or parsed stay [`LinusError::Basis`].
pub(crate) fn basis_error(e: BasisError) -> LinusError {
    match e {
        BasisError::NotFound { name, searched, .. } => {
            let names = available_basis_sets(&searched);
            let hint = did_you_mean(&name, &names);
            let where_ = if searched.is_empty() {
                "an empty basis set search path".to_string()
            } else {
                searched
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            LinusError::Input(format!(
                "unknown basis set '{name}' (no file in {where_}).{hint} linus.available_basis_sets() and `linus --list-basis` list them"
            ))
        }
        BasisError::MissingElement { basis, z } => LinusError::Input(missing_element(&basis, z)),
        e @ BasisError::Unsupported(_) => LinusError::Unsupported(e.to_string()),
        e => LinusError::Basis(e),
    }
}

impl From<BasisError> for LinusError {
    /// The mapping of `basis_error` (did-you-mean hints, element symbols, the kinds of
    /// unsupported content), so that `?` on a basis set error gives the actionable message.
    fn from(e: BasisError) -> Self {
        basis_error(e)
    }
}

fn missing_element(basis: &str, z: u32) -> String {
    let sym = linus_core::elements::z_to_symbol(z).unwrap_or("?");
    format!(
        "basis set {basis:?} has no functions for element {sym} (Z={z}); the def2 sets (def2-svp, def2-tzvp, def2-qzvp, ...) cover H-Rn, \
         with ECPs from Rb on"
    )
}
