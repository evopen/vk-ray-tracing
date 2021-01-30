use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "src/engine/shaders/target"]
pub(super) struct Shaders;
