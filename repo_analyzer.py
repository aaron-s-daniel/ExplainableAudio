import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
import anthropic
from git import Repo
import argparse

# Load environment variables from .env file
load_dotenv()

class ClaudeRepoAnalyzer:
    def __init__(
        self, 
        api_key: str = None,
        output_dir: str = "analysis_results",
        file_types: List[str] = None,
        exclude_dirs: List[str] = None
    ):
        """Initialize the repository analyzer."""
        self.repo_path = os.getcwd()  # Use current working directory
        # Use API key from .env if not provided
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided. Set ANTHROPIC_API_KEY in .env or provide via --api-key")
            
        self.client = anthropic.Client(api_key=self.api_key)
        self.output_dir = output_dir
        self.file_types = file_types or ['.py']
        self.exclude_dirs = exclude_dirs or ['.git', '__pycache__', 'venv', 'env']
        
        # Create output directory BEFORE setting up logging
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging after creating directory
        self.setup_logging()
        
        self.repo = Repo(self.repo_path)

    def setup_logging(self):
        """Configure logging for the analyzer."""
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        log_file = os.path.join(self.output_dir, 'analysis.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_files_to_analyze(self) -> List[str]:
        """Get list of files to analyze based on file types and exclusions."""
        files_to_analyze = []
        
        for root, dirs, files in os.walk(self.repo_path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in self.file_types):
                    file_path = os.path.join(root, file)
                    files_to_analyze.append(file_path)
        
        return files_to_analyze

    def analyze_file(self, file_path: str) -> Dict:
        """Analyze a single file using Claude."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            relative_path = os.path.relpath(file_path, self.repo_path)
            self.logger.info(f"Analyzing {relative_path}")

            # Get analysis from Claude
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": f"""Please analyze this code file and provide a structured analysis with the following sections:

1. Code Quality Assessment:
   - Overall code structure and organization
   - Adherence to style guides
   - Code complexity and maintainability
   - Potential code smells

2. Security Analysis:
   - Potential security vulnerabilities
   - Use of unsafe functions or practices
   - Data validation issues
   - Authentication/authorization concerns

3. Performance Optimization:
   - Performance bottlenecks
   - Memory usage concerns
   - Optimization opportunities
   - Algorithmic efficiency

4. Documentation & Testing:
   - Documentation completeness
   - Code comments quality
   - Test coverage assessment
   - API documentation

5. Specific Recommendations:
   - Concrete suggestions for improvement
   - Code refactoring opportunities
   - Modern alternatives to used approaches
   - Best practices implementation

Please provide specific examples and line numbers where applicable.

Code to analyze:
```
{content}
```"""
                }]
            )

            analysis = response.content[0].text

            return {
                'file_path': relative_path,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat(),
                'file_size': len(content),
                'lines_of_code': len(content.splitlines())
            }

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}")
            return {
                'file_path': relative_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def save_analysis(self, analysis_results: List[Dict]):
        """Save analysis results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(
            self.output_dir, 
            f'analysis_results_{timestamp}.json'
        )
        
        with open(output_file, 'w') as f:
            json.dump(
                {
                    'repository': self.repo_path,
                    'analysis_date': datetime.now().isoformat(),
                    'results': analysis_results
                },
                f,
                indent=2
            )
        
        self.logger.info(f"Analysis results saved to {output_file}")

    def generate_report(self, analysis_results: List[Dict]):
        """Generate a markdown report from analysis results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(
            self.output_dir, 
            f'analysis_report_{timestamp}.md'
        )
        
        with open(report_file, 'w') as f:
            f.write(f"# Repository Analysis Report\n\n")
            f.write(f"Repository: {self.repo_path}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for result in analysis_results:
                f.write(f"## {result['file_path']}\n\n")
                if 'error' in result:
                    f.write(f"Error during analysis: {result['error']}\n\n")
                else:
                    f.write(result['analysis'])
                    f.write("\n\n---\n\n")
        
        self.logger.info(f"Analysis report generated at {report_file}")

    def analyze_repository(self):
        """Analyze the entire repository."""
        self.logger.info(f"Starting analysis of repository: {self.repo_path}")
        
        files = self.get_files_to_analyze()
        self.logger.info(f"Found {len(files)} files to analyze")
        
        analysis_results = []
        for file_path in files:
            result = self.analyze_file(file_path)
            analysis_results.append(result)
        
        self.save_analysis(analysis_results)
        self.generate_report(analysis_results)
        
        self.logger.info("Repository analysis completed")
        
        return analysis_results

def main():
    parser = argparse.ArgumentParser(
        description='Analyze a GitHub repository using Claude'
    )
    parser.add_argument(
        '--api-key',
        help='Anthropic API key (optional if set in .env file)'
    )
    parser.add_argument(
        '--output-dir',
        default='analysis_results',
        help='Directory to store analysis results'
    )
    parser.add_argument(
        '--file-types',
        nargs='+',
        default=['.py'],
        help='File types to analyze (e.g., .py .js .cpp)'
    )
    parser.add_argument(
        '--exclude-dirs',
        nargs='+',
        default=['.git', '__pycache__', 'venv', 'env'],
        help='Directories to exclude from analysis'
    )
    
    args = parser.parse_args()
    
    analyzer = ClaudeRepoAnalyzer(
        api_key=args.api_key,
        output_dir=args.output_dir,
        file_types=args.file_types,
        exclude_dirs=args.exclude_dirs
    )
    
    analyzer.analyze_repository()

if __name__ == "__main__":
    main()