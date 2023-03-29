use crate::error::ShaderCompileError::{self, *};
use ash::util::read_spv;
use bitflags::bitflags;
use derive_more::Into;
use once_cell::sync::Lazy;
use shaderc::Compiler;
pub use shaderc::ShaderKind;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

#[derive(Into)]
pub struct ShaderCode(Vec<u32>);

static SHADERC_COMPILER: Lazy<Compiler> = Lazy::new(|| Compiler::new().unwrap());

#[macro_export]
macro_rules! shader {
    ($path: literal $(, $options: expr)?) => {
        $crate::macros::get_shader_code($path, include_str!($path), ($($options)?).into(), std::path::Path::new(file!()).parent().unwrap().into())
    };
    ($path: expr $(, $options: expr)?) => {
        match std::fs::read($path) {
            Ok(text) => {
                $crate::macros::get_shader_code($path, std::str::from_utf8(&text).unwrap(), ($($options)?).into(), std::env::current_dir().unwrap())
            },
            Err(err) => {
                Err($crate::error::ShaderCompileError::from(err))
            }
        }
    };
}

pub fn get_shader_code(
    path: &str,
    text: &str,
    options: ShaderOptions,
    invocation_path: PathBuf,
) -> Result<ShaderCode, ShaderCompileError> {
    let cache_enabled = options.contains(ShaderOptions::CACHE) && cfg!(target_os = "linux");
    let flat_path = String::from(path).replace("/", "_");
    let spirv_path = String::from("/tmp/silt_") + &flat_path + ".spirv";
    let copy_path = String::from("/tmp/silt_") + &flat_path;

    if cache_enabled {
        let spirv_file = fs::File::open(&spirv_path);
        let copy_text = fs::read_to_string(&copy_path);

        match (spirv_file, copy_text) {
            (Ok(ref mut spirv_file), Ok(copy_text)) if copy_text == text => {
                let code = read_spv(spirv_file).map_err(|_| AshLoadFailure)?;
                return Ok(ShaderCode(code));
            }
            _ => (),
        }
    }

    let shader_kind = get_kind(path).ok_or(ShaderKindUnknown(path.into()))?;
    let spirv = SHADERC_COMPILER
        .compile_into_spirv(text, shader_kind, path, "main", None)
        .map_err(Into::<ShaderCompileError>::into)?;
    let code = read_spv(&mut Cursor::new(spirv.as_binary_u8())).map_err(|_| AshLoadFailure)?;

    if cache_enabled {
        fs::write(&spirv_path, spirv.as_binary_u8()).unwrap();
        fs::copy(invocation_path.join(path), copy_path).unwrap();
    }

    Ok(ShaderCode(code))
}

pub fn get_kind(path: &str) -> Option<shaderc::ShaderKind> {
    let extension = Path::new(path).extension()?.to_str()?;
    match extension {
        "vert" => Some(shaderc::ShaderKind::Vertex),
        "frag" => Some(shaderc::ShaderKind::Fragment),
        "comp" => Some(shaderc::ShaderKind::Compute),
        _ => None,
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ShaderOptions: u32 {
        const CACHE = 0b00000001;
    }
}

impl Into<ShaderOptions> for () {
    fn into(self) -> ShaderOptions {
        ShaderOptions::empty()
    }
}
