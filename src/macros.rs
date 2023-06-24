use anyhow::{anyhow, Result};
use ash::util::read_spv;
use once_cell::sync::Lazy;
use shaderc::{CompileOptions, Compiler, ShaderKind};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use crate::material::{ShaderOptions};

static SHADERC_COMPILER: Lazy<Compiler> = Lazy::new(|| Compiler::new().unwrap());

#[macro_export]
macro_rules! compile {
    ($path: literal, $options: expr) => {
        $crate::macros::__get_shader_code($path, include_str!($path), ($options).into(), std::path::Path::new(file!()).parent().unwrap().into())
    };
    ($path: expr, $options: expr) => {
        match std::fs::read($path) {
            Ok(text) => {
                $crate::macros::__get_shader_code($path, std::str::from_utf8(&text).unwrap(), ($options).into(), std::env::current_dir().unwrap())
            },
            _ => Err(anyhow::anyhow!("failed to read shader code"))
        }
    };
}

pub fn __get_shader_code(
    path: &str,
    text: &str,
    options: ShaderOptions,
    invocation_path: PathBuf,
) -> Result<(Vec<u32>, ShaderKind)> {
    let cache_enabled = options.contains(ShaderOptions::CACHE) && cfg!(target_os = "linux");
    let compile_options = if options.contains(ShaderOptions::HLSL) {
        let mut options = CompileOptions::new().unwrap();
        options.set_source_language(shaderc::SourceLanguage::HLSL);
        Some(options)
    } else {
        None
    };

    let kind = get_kind(path).ok_or(anyhow!("failed to determine shader type"))?;

    let flat_path = String::from(path).replace("/", "_");
    let spirv_path = String::from("/tmp/silt_") + &flat_path + ".spirv";
    let copy_path = String::from("/tmp/silt_") + &flat_path;

    if cache_enabled {
        let spirv_file = fs::File::open(&spirv_path);
        let copy_text = fs::read_to_string(&copy_path);

        match (spirv_file, copy_text) {
            (Ok(ref mut spirv_file), Ok(copy_text)) if copy_text == text => {
                let code = read_spv(spirv_file)?;
                return Ok((code, kind));
            }
            _ => (),
        }
    }

    let spirv =
        SHADERC_COMPILER.compile_into_spirv(text, kind, path, "main", compile_options.as_ref())?;
    let code = read_spv(&mut Cursor::new(spirv.as_binary_u8()))?;

    if cache_enabled {
        fs::write(&spirv_path, spirv.as_binary_u8())?;
        fs::copy(invocation_path.join(path), copy_path)?;
    }

    Ok((code, kind))
}

fn get_kind(path: &str) -> Option<shaderc::ShaderKind> {
    let extension = Path::new(path).extension()?.to_str()?;
    match extension {
        "vert" => Some(shaderc::ShaderKind::Vertex),
        "frag" => Some(shaderc::ShaderKind::Fragment),
        "comp" => Some(shaderc::ShaderKind::Compute),
        _ => None,
    }
}

#[macro_export]
macro_rules! bindable {
    ($name: ident, $ty: expr, $frequency: expr, $binding: expr) => {
        #[derive(Default, Copy, Clone)]
        pub struct $name();

        impl $crate::storage::descriptors::Bindable for $name {
            fn binding(&self) -> BindingDescription {
                BindingDescription {
                    ty: $ty,
                    frequency: $frequency,
                    binding: $binding,
                    ..Default::default()
                }
            }
        }
    };
}
