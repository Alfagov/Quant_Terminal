# QuantSim Terminal

QuantSim Terminal is a high-performance, quantitative finance simulation platform built with Rust and modern web technologies. It provides interactive tools for visualizing and analyzing complex financial models, market-making strategies, and stochastic processes.

## 🚀 Features

- **Advanced Financial Models**:
  - **Avellaneda-Stoikov**: Market-making strategy simulation with inventory risk and reservation prices.
  - **Black-Scholes**: Option pricing model with real-time "Greeks" calculation and 3D volatility surface visualization.
  - **Glosten-Milgrom**: Information-based market-making model simulating informed vs. uninformed order flow.
  - **Geometric Brownian Motion (GBM)**: Stochastic process simulation for asset price paths with configurable drift and volatility.
  - **Binomial Poisson**: Jump-diffusion process modeling.
  - **Option Strategies**: Payoff diagrams and analysis for multi-leg option strategies (Straddles, Iron Condors, etc.).
  - **Bond Portfolio Analysis**: Yield curve construction and duration/convexity metrics.

- **High-Performance Backend**:
  - Built on **Rust** using the **Axum** web framework for ultra-low latency.
  - Efficient numerical computations using `nalgebra` and `statrs`.
  - Parallel processing support with `rayon`.

- **Modern Cyberpunk UI**:
  - **HTMX** for seamless, partial page updates without full reloads.
  - **Tailwind CSS v4** for a responsive, futuristic "glassmorphism" terminal aesthetic.
  - **Chart.js** for dynamic, interactive financial charting.
  - Server-side rendering with **Askama** templates.

## 🛠️ Tech Stack

- **Backend**: Rust, Axum, Tokio, Tower
- **Frontend**: HTML5, Tailwind CSS, JavaScript, HTMX, Chart.js
- **Templating**: Askama (Jinja-like syntax for Rust)
- **Math/Stats**: `statrs`, `nalgebra`, `rand`

## 📦 Installation

### Prerequisites

- **Rust**: Ensure you have the latest stable version of Rust installed. [Install Rust](https://www.rust-lang.org/tools/install)
- **Node.js & npm**: Required for building Tailwind CSS styles.

### Steps

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/quant_terminal.git
    cd quant_terminal
    ```

2.  **Install CSS dependencies**:
    ```bash
    npm install
    ```

3.  **Build Tailwind CSS**:
    To watch for changes during development:
    ```bash
    npx @tailwindcss/cli -i ./static/css/styles.css -o ./static/css/output.css --watch
    ```

4.  **Run the Rust Server**:
    In a new terminal window:
    ```bash
    cargo run
    ```

5.  **Access the Application**:
    Open your browser and navigate to:
    `http://localhost:8080`

## 🖥️ Usage

1.  **Dashboard**: The landing page provides an overview of available modules. Select a module to begin.
2.  **Parameter Configuration**: Use the sidebar inputs to adjust model parameters (e.g., specific volatility, time horizon, drift).
    - **Tooltips**: Hover over the `(i)` icons next to parameters for detailed explanations.
3.  **Simulation**: Click "Run Simulation" (or similar) to execute the model on the backend.
4.  **Visualization**: Results are rendered instantly in the main view as interactive charts and statistical summaries.

## 📂 Project Structure

- `src/`: Rust backend source code.
    - `handlers/`: HTTP request handlers.
    - `modules/`: Implementation of financial models (Black-Scholes, etc.).
- `static/`: Static assets (CSS, JS, images).
- `templates/`: HTML templates (Askama).
- `Cargo.toml`: Rust dependencies and configuration.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
