// SPDX-License-Identifier: GPL-3.0-or-later

//! First-class PM7-family method selection.

use crate::error::{Pm7Error, Result};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Pm7Method {
    #[default]
    Pm7,
    Pm7Ts,
    Pm7Sparkle,
    Pm7Minus,
    Pm7Hh,
}

impl Pm7Method {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pm7 => "pm7",
            Self::Pm7Ts => "pm7-ts",
            Self::Pm7Sparkle => "pm7-sparkle",
            Self::Pm7Minus => "pm7-minus",
            Self::Pm7Hh => "pm7-hh",
        }
    }

    pub const fn uses_sparkles(self) -> bool {
        matches!(self, Self::Pm7Sparkle)
    }

    pub const fn has_post_scf_corrections(self) -> bool {
        !matches!(self, Self::Pm7Minus)
    }

    pub const fn has_hh_repulsion(self) -> bool {
        matches!(self, Self::Pm7Hh)
    }
}

impl FromStr for Pm7Method {
    type Err = Pm7Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pm7" => Ok(Self::Pm7),
            "pm7-ts" | "pm7ts" => Ok(Self::Pm7Ts),
            "pm7-sparkle" | "pm7+sparkle" | "sparkle" => Ok(Self::Pm7Sparkle),
            "pm7-minus" | "pm7-" | "pm7minus" => Ok(Self::Pm7Minus),
            "pm7-hh" | "pm7hh" => Ok(Self::Pm7Hh),
            _ => Err(Pm7Error::InvalidInput(format!(
                "unknown PM7 method `{value}`"
            ))),
        }
    }
}

impl std::fmt::Display for Pm7Method {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_method_names() {
        assert_eq!("PM7-TS".parse::<Pm7Method>().unwrap(), Pm7Method::Pm7Ts);
        assert_eq!("pm7-".parse::<Pm7Method>().unwrap(), Pm7Method::Pm7Minus);
        assert!(!Pm7Method::Pm7Minus.has_post_scf_corrections());
    }
}
