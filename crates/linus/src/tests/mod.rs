//! The method crates' unit tests that run whole calculations (the Session). They moved here
//! from their crates, which do not depend on the facade; each module says where from.

mod dft_vv10;
mod dft_vv10_resp;
mod jk_df_gradient;
mod jk_df_hessian;
mod jk_df_response;
mod jk_exact_gradient;
mod jk_rij_cosx;
