//! Inkling 2 Mini on MLX: weights, caches, Metal kernels, forward pass and
//! greedy generation.

pub mod cache;
pub mod draft;
pub mod generate;
pub mod kernels;
pub mod model;
pub mod ops;

pub use model::Model;
