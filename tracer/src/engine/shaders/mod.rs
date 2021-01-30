use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "src/engine/shaders/spirv"]
pub(super) struct Shaders;
