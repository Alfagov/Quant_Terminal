use askama::Template;
use serde::Serialize;
use crate::handlers::{AppError, ChartTemplate};

// Chart dataset structure
#[derive(Debug, Clone, Serialize)]
pub struct Dataset {
    pub label: String,
    pub data: String,
    pub border_color: String,
    pub background_color: String,
    pub border_width: u32,
    pub fill: bool,
    pub tension: f64,
    pub border_dash: Option<String>,
}

pub fn create_dataset(
    label: &str,
    data: &[f64],
    border_color: &str,
    background_color: &str,
    border_width: u32,
    fill: bool,
) -> Dataset {
    Dataset {
        label: label.to_string(),
        data: serde_json::to_string(data).unwrap(),
        border_color: border_color.to_string(),
        background_color: background_color.to_string(),
        border_width,
        fill,
        tension: 0.1,
        border_dash: None,
    }
}

pub fn create_dataset_with_dash(
    label: &str,
    data: &[f64],
    border_color: &str,
    background_color: &str,
    border_width: u32,
    fill: bool,
    dash: Vec<u32>,
) -> Dataset {
    let mut ds = create_dataset(label, data, border_color, background_color, border_width, fill);
    ds.border_dash = Some(serde_json::to_string(&dash).unwrap());
    ds
}

pub fn render_chart<T: Serialize>(
    chart_id: &str,
    title: &str,
    labels: &[T],
    datasets: Vec<Dataset>,
    x_axis_label: &str,
    y_axis_label: &str,
    legend_display: bool,
) -> Result<String, AppError> {
    let chart = ChartTemplate {
        chart_id: chart_id.to_string(),
        title: title.to_string(),
        labels: serde_json::to_string(labels).unwrap(),
        datasets,
        x_axis_label: x_axis_label.to_string(),
        y_axis_label: y_axis_label.to_string(),
        legend_display,
    };

    Ok(chart.render()?)
}