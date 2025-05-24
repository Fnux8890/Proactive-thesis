use chrono::{DateTime, NaiveDateTime, Utc};
use chrono_tz::Tz;

/// Parse a float that may use comma as decimal separator
pub fn parse_locale_float(s: &str) -> Result<f64, std::num::ParseFloatError> {
    s.replace(',', ".").parse::<f64>()
}

/// Parse a local datetime string with optional timezone information
pub fn parse_datetime_with_tz(
    datetime_str: &str,
    format: &str,
    tz: Option<&Tz>,
) -> Result<DateTime<Utc>, String> {
    let naive = NaiveDateTime::parse_from_str(datetime_str, format).map_err(|e| {
        format!(
            "Failed to parse timestamp '{}' with format '{}': {}",
            datetime_str, format, e
        )
    })?;
    if let Some(tz) = tz {
        tz.from_local_datetime(&naive)
            .single()
            .ok_or_else(|| {
                format!(
                    "Ambiguous or invalid local time '{}' for timezone {}",
                    datetime_str, tz
                )
            })
            .map(|dt| dt.with_timezone(&Utc))
    } else {
        Ok(Utc.from_utc_datetime(&naive))
    }
}
