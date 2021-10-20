use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

fn main() -> anyhow::Result<()> {
    let extensions = &[];

    let capabilities = &[Capability::ImageQuery];

    compile_shader_multi("shaders/rendering", extensions, capabilities)?;

    Ok(())
}

fn compile_shader_multi(
    path: &str,
    extensions: &[&str],
    capabilities: &[Capability],
) -> anyhow::Result<()> {
    let mut builder = SpirvBuilder::new(path, "spirv-unknown-spv1.1")
        .print_metadata(MetadataPrintout::None)
        .multimodule(true);

    for extension in extensions {
        builder = builder.extension(*extension);
    }

    for capability in capabilities {
        builder = builder.capability(*capability);
    }

    let result = builder.build()?;

    for (name, path) in result.module.unwrap_multi() {
        std::fs::copy(path, &format!("shaders/{}.spv", name))?;
    }

    Ok(())
}
