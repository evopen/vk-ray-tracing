use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "../target/spirv"]
pub(super) struct Shaders;
