use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShaderCompileError {
    #[error("shaderc compiler failed to initialize")]
    CompilerInitializationFailure,
    #[error("unable to determine type of shader '{0}'")]
    ShaderKindUnknown(String),
    #[error("failure to compile shader to spirv")]
    ShadercCompileFailure(#[from] shaderc::Error),
    #[error("failure to load spirv file in ash")]
    AshLoadFailure,
    #[error("could not find shader file")]
    FileNotFound(#[from] std::io::Error)
}