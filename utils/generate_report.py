import json
import os
from datetime import datetime

def generate_html_report():
    # Read the latest test run results
    results_file = ".deepeval/.latest_test_run.json"
    
    if not os.path.exists(results_file):
        print(f"No results found at {results_file}")
        return
    
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_run = data.get("testRunData", {})
    test_cases = test_run.get("testCases", [])
    metrics_scores = test_run.get("metricsScores", [])
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DeepEval Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .metrics-overview {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            border-bottom: 1px solid #eee;
        }}
        .metric-item:last-child {{
            border-bottom: none;
        }}
        .metric-name {{
            font-weight: bold;
            color: #333;
        }}
        .metric-score {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        .score-good {{ color: #28a745; }}
        .score-warning {{ color: #ffc107; }}
        .score-bad {{ color: #dc3545; }}
        .test-case {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .test-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #eee;
        }}
        .test-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        .badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}
        .metric-details {{
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .metric-result {{
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-left: 4px solid #667eea;
            border-radius: 3px;
        }}
        .reason {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
            line-height: 1.5;
        }}
        .cost {{
            color: #999;
            font-size: 0.85em;
        }}
        .section-title {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 DeepEval Test Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Test File: {test_run.get('testFile', 'N/A')}</p>
    </div>
    
    <div class="summary">
        <h2 class="section-title">📊 Summary</h2>
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{test_run.get('testPassed', 0)}</div>
                <div class="stat-label">Tests Passed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{test_run.get('testFailed', 0)}</div>
                <div class="stat-label">Tests Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{test_run.get('runDuration', 0):.2f}s</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${test_run.get('evaluationCost', 0):.4f}</div>
                <div class="stat-label">Evaluation Cost</div>
            </div>
        </div>
    </div>
    
    <div class="metrics-overview">
        <h2 class="section-title">📈 Metrics Overview</h2>
"""
    
    # Add metrics overview
    for metric in metrics_scores:
        avg_score = sum(metric.get('scores', [])) / len(metric.get('scores', [1])) if metric.get('scores') else 0
        score_class = "score-good" if avg_score >= 0.7 else "score-warning" if avg_score >= 0.5 else "score-bad"
        
        html += f"""
        <div class="metric-item">
            <div>
                <div class="metric-name">{metric.get('metric', 'Unknown')}</div>
                <div class="cost">Passes: {metric.get('passes', 0)} | Fails: {metric.get('fails', 0)}</div>
            </div>
            <div class="metric-score {score_class}">{avg_score:.2f}</div>
        </div>
"""
    
    html += """
    </div>
    
    <h2 class="section-title">🔍 Test Cases</h2>
"""
    
    # Add test cases
    for i, test_case in enumerate(test_cases, 1):
        success = test_case.get('success', False)
        badge_class = "badge-success" if success else "badge-danger"
        badge_text = "✓ PASSED" if success else "✗ FAILED"
        
        html += f"""
    <div class="test-case">
        <div class="test-header">
            <div class="test-name">Test {i}: {test_case.get('name', 'Unnamed')}</div>
            <div class="badge {badge_class}">{badge_text}</div>
        </div>
        
        <div><strong>Duration:</strong> {test_case.get('runDuration', 0):.2f}s</div>
        <div><strong>Cost:</strong> ${test_case.get('evaluationCost', 0):.4f}</div>
        
        <div class="metric-details">
            <h3>Metrics:</h3>
"""
        
        # Add metric results
        for metric in test_case.get('metricsData', []):
            metric_success = metric.get('success', False)
            metric_score = metric.get('score', 0)
            score_class = "score-good" if metric_score >= 0.7 else "score-warning" if metric_score >= 0.5 else "score-bad"
            
            html += f"""
            <div class="metric-result">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>{metric.get('name', 'Unknown Metric')}</strong>
                    <span class="metric-score {score_class}">{metric_score:.2f}</span>
                </div>
                <div><em>Threshold: {metric.get('threshold', 0)}</em></div>
                <div class="reason">{metric.get('reason', 'No reason provided')}</div>
                <div class="cost">Model: {metric.get('evaluationModel', 'N/A')} | Cost: ${metric.get('evaluationCost', 0):.6f}</div>
            </div>
"""
        
        html += """
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    # Save the report
    output_file = "test_report.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_file}")
    print(f"📂 Open it in your browser to view the results")
    
    return output_file

if __name__ == "__main__":
    generate_html_report()
