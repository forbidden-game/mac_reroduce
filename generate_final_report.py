#!/usr/bin/env python3
"""
MAC Comprehensive Experiments Final Report Generator

This script generates a comprehensive HTML report with interactive visualizations
for the MAC experiment results.
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

class MACReportGenerator:
    """Generate comprehensive HTML reports for MAC experiments"""
    
    def __init__(self, config_path: str = "automation_config.yaml"):
        self.config = self.load_config(config_path)
        self.experiment_dir = Path(self.config['experiment_settings']['output_dir'])
        self.analysis_dir = self.experiment_dir / "cross_dataset_analysis"
        self.report_dir = self.experiment_dir / "final_report"
        self.report_dir.mkdir(exist_ok=True)
        
        print("📝 MAC Report Generator initialized")
        print(f"📁 Analysis directory: {self.analysis_dir}")
        print(f"📄 Report output: {self.report_dir}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    
    def load_analysis_results(self) -> Dict[str, Any]:
        """Load all analysis results"""
        results = {}
        
        # Load raw results
        raw_results_file = self.analysis_dir / "raw_results.json"
        if raw_results_file.exists():
            with open(raw_results_file, 'r') as f:
                results['raw_results'] = json.load(f)
        
        # Load performance summary
        summary_file = self.analysis_dir / "performance_summary.csv"
        if summary_file.exists():
            results['performance_df'] = pd.read_csv(summary_file)
        
        # Load best models
        best_models_file = self.analysis_dir / "best_models_summary.json"
        if best_models_file.exists():
            with open(best_models_file, 'r') as f:
                results['best_models'] = json.load(f)
        
        # Load statistical analysis
        stats_file = self.analysis_dir / "statistical_analysis.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                results['statistics'] = json.load(f)
        
        # Load summary statistics
        summary_stats_file = self.analysis_dir / "summary_statistics.json"
        if summary_stats_file.exists():
            with open(summary_stats_file, 'r') as f:
                results['summary_stats'] = json.load(f)
        
        return results
    
    def encode_image_to_base64(self, image_path: Path) -> str:
        """Encode image to base64 for HTML embedding"""
        try:
            with open(image_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            print(f"⚠️  Error encoding image {image_path}: {e}")
            return ""
    
    def create_interactive_plots(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Create interactive Plotly visualizations"""
        plots = {}
        
        if 'performance_df' not in results:
            return plots
        
        df = results['performance_df']
        
        # 1. SNR vs Performance Interactive Plot
        fig_snr = go.Figure()
        
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            fig_snr.add_trace(go.Scatter(
                x=dataset_df['SNR'],
                y=dataset_df['Best_Val_Loss'],
                mode='lines+markers',
                name=dataset,
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'SNR: %{x} dB<br>' +
                             'Best Val Loss: %{y:.4f}<br>' +
                             '<extra></extra>'
            ))
        
        fig_snr.update_layout(
            title='MAC Performance vs SNR - Interactive View',
            xaxis_title='SNR (dB)',
            yaxis_title='Best Validation Loss',
            hovermode='closest',
            template='plotly_white',
            width=800,
            height=500
        )
        
        plots['snr_performance'] = pyo.plot(fig_snr, include_plotlyjs=False, output_type='div')
        
        # 2. Training Time Analysis
        fig_time = go.Figure()
        
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            fig_time.add_trace(go.Bar(
                x=dataset_df['SNR'],
                y=dataset_df['Training_Time_Hours'],
                name=dataset,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'SNR: %{x} dB<br>' +
                             'Training Time: %{y:.2f} hours<br>' +
                             '<extra></extra>'
            ))
        
        fig_time.update_layout(
            title='Training Time by Dataset and SNR',
            xaxis_title='SNR (dB)',
            yaxis_title='Training Time (hours)',
            template='plotly_white',
            width=800,
            height=500
        )
        
        plots['training_time'] = pyo.plot(fig_time, include_plotlyjs=False, output_type='div')
        
        # 3. Performance Distribution
        fig_dist = go.Figure()
        
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            fig_dist.add_trace(go.Box(
                y=dataset_df['Best_Val_Loss'],
                name=dataset,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig_dist.update_layout(
            title='Performance Distribution by Dataset',
            yaxis_title='Best Validation Loss',
            template='plotly_white',
            width=600,
            height=500
        )
        
        plots['performance_distribution'] = pyo.plot(fig_dist, include_plotlyjs=False, output_type='div')
        
        return plots
    
    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        
        # Create interactive plots
        interactive_plots = self.create_interactive_plots(results)
        
        # Load static images
        static_images = {}
        image_files = [
            'snr_sensitivity_analysis.png',
            'convergence_analysis.png'
        ]
        
        for img_file in image_files:
            img_path = self.analysis_dir / img_file
            if img_path.exists():
                static_images[img_file] = self.encode_image_to_base64(img_path)
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAC Comprehensive Experiments Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #2c3e50;
            margin-top: 25px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: white;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .dataset-section {{
            background-color: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }}
        .best-model {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .static-image {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-style: italic;
            margin-top: 20px;
        }}
        .recommendation {{
            background-color: #e8f5e8;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 10px 0;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 MAC Comprehensive Experiments Report</h1>
        
        <div class="timestamp">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        <h2>📊 Executive Summary</h2>
        <div class="summary-grid">
        """
        
        # Add summary cards
        if 'performance_df' in results:
            df = results['performance_df']
            total_experiments = len(df)
            completed_experiments = df['Completed'].sum()
            avg_training_time = df['Training_Time_Hours'].mean()
            best_performance = df['Best_Val_Loss'].min()
            
            html_content += f"""
            <div class="summary-card">
                <h3>Total Experiments</h3>
                <div class="value">{total_experiments}</div>
            </div>
            <div class="summary-card">
                <h3>Success Rate</h3>
                <div class="value">{completed_experiments/total_experiments*100:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Best Performance</h3>
                <div class="value">{best_performance:.4f}</div>
            </div>
            <div class="summary-card">
                <h3>Avg Training Time</h3>
                <div class="value">{avg_training_time:.1f}h</div>
            </div>
            """
        
        html_content += """
        </div>
        
        <h2>🏆 Best Models</h2>
        """
        
        # Add best models section
        if 'best_models' in results:
            best_models = results['best_models']
            
            if 'overall_best' in best_models:
                best = best_models['overall_best']
                html_content += f"""
                <div class="best-model">
                    <h3>🥇 Overall Best Model</h3>
                    <p><strong>Dataset:</strong> {best['dataset']}</p>
                    <p><strong>SNR:</strong> {best['snr']} dB</p>
                    <p><strong>Validation Loss:</strong> {best['loss']:.4f}</p>
                </div>
                """
            
            if 'per_dataset' in best_models:
                html_content += "<h3>Best Models per Dataset</h3>"
                for dataset, best in best_models['per_dataset'].items():
                    html_content += f"""
                    <div class="best-model">
                        <strong>{dataset}:</strong> SNR {best['snr']}dB (Loss: {best['loss']:.4f})
                    </div>
                    """
        
        # Add interactive plots
        html_content += """
        <h2>📈 Interactive Performance Analysis</h2>
        """
        
        for plot_name, plot_div in interactive_plots.items():
            html_content += f"""
            <div class="plot-container">
                {plot_div}
            </div>
            """
        
        # Add static analysis images
        html_content += """
        <h2>📊 Detailed Analysis</h2>
        <h3>SNR Sensitivity Analysis</h3>
        """
        
        if 'snr_sensitivity_analysis.png' in static_images:
            html_content += f"""
            <div class="plot-container">
                <img src="{static_images['snr_sensitivity_analysis.png']}" 
                     alt="SNR Sensitivity Analysis" class="static-image">
            </div>
            """
        
        html_content += "<h3>Convergence Analysis</h3>"
        if 'convergence_analysis.png' in static_images:
            html_content += f"""
            <div class="plot-container">
                <img src="{static_images['convergence_analysis.png']}" 
                     alt="Convergence Analysis" class="static-image">
            </div>
            """
        
        # Add dataset details
        if 'summary_stats' in results:
            html_content += """
            <h2>📋 Dataset Performance Details</h2>
            """
            
            for dataset, stats in results['summary_stats'].items():
                html_content += f"""
                <div class="dataset-section">
                    <h3>{dataset}</h3>
                    <p><strong>Experiments:</strong> {stats['num_experiments']} 
                       ({stats['completed_experiments']} completed)</p>
                    <p><strong>Average Validation Loss:</strong> {stats['avg_best_val_loss']:.4f} ± {stats['std_best_val_loss']:.4f}</p>
                    <p><strong>Total Training Time:</strong> {stats['total_training_time']:.2f} hours</p>
                    <p><strong>Best SNR:</strong> {stats['best_snr_performance']['snr']} dB 
                       (Loss: {stats['best_snr_performance']['loss']:.4f})</p>
                </div>
                """
        
        # Add recommendations
        html_content += """
        <h2>💡 Recommendations</h2>
        <div class="recommendation">
            <h3>Model Selection</h3>
            <p>Based on the analysis, we recommend using the overall best model for optimal performance. 
               Consider the SNR range of your deployment environment when selecting models.</p>
        </div>
        
        <div class="recommendation">
            <h3>Training Efficiency</h3>
            <p>Training convergence analysis suggests that early stopping could be implemented to reduce 
               training time without significant performance loss.</p>
        </div>
        
        <div class="warning">
            <h3>Important Considerations</h3>
            <p>Performance varies significantly across SNR levels. Ensure your deployment SNR range 
               matches the training conditions for optimal results.</p>
        </div>
        
        <h2>📈 Statistical Analysis</h2>
        """
        
        # Add statistical results
        if 'statistics' in results:
            stats = results['statistics']
            if 'anova_test' in stats:
                anova = stats['anova_test']
                html_content += f"""
                <p><strong>ANOVA Test:</strong> F-statistic = {anova.get('f_statistic', 'N/A'):.4f}, 
                   p-value = {anova.get('p_value', 'N/A'):.4f}</p>
                <p><strong>Interpretation:</strong> {anova.get('interpretation', 'No analysis available')}</p>
                """
        
        # Close HTML
        html_content += """
        <div class="timestamp">
            <p>This report was automatically generated by the MAC Comprehensive Experiments system.</p>
            <p>For questions or issues, please refer to the analysis logs and raw data files.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary PDF content"""
        summary = []
        summary.append("MAC COMPREHENSIVE EXPERIMENTS - EXECUTIVE SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        if 'performance_df' in results:
            df = results['performance_df']
            summary.append("KEY FINDINGS:")
            summary.append(f"• Total experiments conducted: {len(df)}")
            summary.append(f"• Success rate: {df['Completed'].sum()/len(df)*100:.1f}%")
            summary.append(f"• Best validation loss achieved: {df['Best_Val_Loss'].min():.4f}")
            summary.append(f"• Average training time: {df['Training_Time_Hours'].mean():.2f} hours")
            summary.append("")
        
        if 'best_models' in results and 'overall_best' in results['best_models']:
            best = results['best_models']['overall_best']
            summary.append("BEST MODEL:")
            summary.append(f"• Dataset: {best['dataset']}")
            summary.append(f"• SNR: {best['snr']} dB")
            summary.append(f"• Validation Loss: {best['loss']:.4f}")
            summary.append("")
        
        summary.append("RECOMMENDATIONS:")
        summary.append("• Deploy the overall best model for optimal performance")
        summary.append("• Consider SNR-specific models for deployment environments")
        summary.append("• Implement early stopping based on convergence analysis")
        summary.append("• Regular retraining recommended for changing environments")
        
        return "\n".join(summary)
    
    def generate_report(self):
        """Generate complete report package"""
        print("📝 Generating comprehensive report...")
        
        # Load analysis results
        results = self.load_analysis_results()
        
        if not results:
            print("❌ No analysis results found. Please run analysis first.")
            return
        
        # Generate HTML report
        html_content = self.generate_html_report(results)
        html_path = self.report_dir / "comprehensive_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTML report saved to {html_path}")
        
        # Generate executive summary
        summary_content = self.generate_executive_summary(results)
        summary_path = self.report_dir / "executive_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        print(f"✅ Executive summary saved to {summary_path}")
        
        # Copy raw data for reference
        raw_data_path = self.report_dir / "raw_analysis_data.json"
        with open(raw_data_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✅ Raw data saved to {raw_data_path}")
        
        print(f"\n🎉 Complete report package generated!")
        print(f"📁 Report directory: {self.report_dir}")
        print(f"🌐 Open {html_path} in your browser to view the interactive report")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MAC experiment final report")
    parser.add_argument("--config", default="automation_config.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    generator = MACReportGenerator(args.config)
    generator.generate_report()


if __name__ == "__main__":
    main()
