pub use shaderc::ShaderKind;
use shaderc::Compiler;
use ash::util::read_spv;
use std::path::Path;
use std::io::Cursor;
use derive_more::Into;

#[derive(Into)]
pub struct ShaderCode(Vec<u32>);

#[macro_export]
macro_rules! shader {
    ($path: literal)=> {
        $crate::macros::get_shader_code($path, include_str!($path)).unwrap()
    };
    ($path: expr)=> {
        {
            let text = String::from_utf8(std::fs::read($path).unwrap()).unwrap();
            $crate::macros::get_shader_code($path.into(), &text).unwrap()
        }
    };
}

pub fn get_shader_code(path: &str, text: &str) -> Option<ShaderCode> {
    let compiler = Compiler::new()?;
    let shader_kind = get_kind(path)?;
    let spirv = compiler.compile_into_spirv(text, shader_kind, path, "main", None).ok()?;
    let code = read_spv(&mut Cursor::new(spirv.as_binary_u8())).ok()?;
    Some(ShaderCode(code))
}

pub fn get_kind(path: &str) -> Option<shaderc::ShaderKind> {
    let extension = Path::new(path).extension()?.to_str()?;
    match extension {
        "vert" => Some(shaderc::ShaderKind::Vertex),
        "frag" => Some(shaderc::ShaderKind::Fragment),
        "comp" => Some(shaderc::ShaderKind::Compute),
        _ => None
    }
}