import streamlit as st
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

class GitHubAPI:
    """Class to handle GitHub API interactions"""
    
    def __init__(self):
        self.GITHUB_API_BASE = "https://api.github.com"
    
    def parse_repo_url(self, repo_url):
        """Parse GitHub repository URL to extract owner and repo name"""
        try:
            parts = repo_url.rstrip("/").split("/")
            owner = parts[-2]
            repo = parts[-1]
            return owner, repo
        except Exception as e:
            st.error(f"Invalid GitHub repository URL: {e}")
            return None, None
    
    def fetch_repo_metadata(self, repo_url):
        """Fetch repository metadata from GitHub API"""
        try:
            owner, repo = self.parse_repo_url(repo_url)
            if not owner or not repo:
                return None
                
            url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "name": data.get("name"),
                    "description": data.get("description"),
                    "stars": data.get("stargazers_count"),
                    "forks": data.get("forks_count"),
                    "open_issues": data.get("open_issues_count"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "language": data.get("language"),
                    "size": data.get("size"),
                    "watchers": data.get("watchers_count"),
                    "default_branch": data.get("default_branch"),
                    "license": data.get("license", {}).get("name") if data.get("license") else "No License",
                    "topics": data.get("topics", [])
                }
            elif response.status_code == 403:
                st.error("Rate limit exceeded. Please try again later.")
            elif response.status_code == 404:
                st.error("Repository not found. Please check the URL.")
            else:
                st.error(f"Failed to fetch repository metadata. HTTP Status Code: {response.status_code}")
            return None
        except Exception as e:
            st.error(f"Error fetching repository metadata: {e}")
            return None
    
    def fetch_contributors(self, repo_url):
        """Fetch repository contributors from GitHub API"""
        try:
            owner, repo = self.parse_repo_url(repo_url)
            if not owner or not repo:
                return None
                
            url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/contributors"
            response = requests.get(url)
            
            if response.status_code == 200:
                contributors = response.json()
                return contributors
            elif response.status_code == 403:
                st.error("Rate limit exceeded. Please try again later.")
            elif response.status_code == 404:
                st.error("Contributors data not found.")
            else:
                st.error(f"Failed to fetch contributors. HTTP Status Code: {response.status_code}")
            return None
        except Exception as e:
            st.error(f"Error fetching contributors: {e}")
            return None
    
    def fetch_commit_activity(self, repo_url):
        """Fetch commit activity data from GitHub API"""
        try:
            owner, repo = self.parse_repo_url(repo_url)
            if not owner or not repo:
                return None
                
            # Get commit activity
            activity_url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/stats/commit_activity"
            activity_response = requests.get(activity_url)
            
            # Get commit history
            commits_url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
            commits_response = requests.get(commits_url)
            
            if activity_response.status_code == 200 and commits_response.status_code == 200:
                activity_data = activity_response.json()
                commits_data = commits_response.json()
                
                # Process commit activity
                weekly_commits = sum([week["total"] for week in activity_data])
                
                # Process commit history
                commit_history = []
                for commit in commits_data:
                    try:
                        commit_date = datetime.strptime(commit['commit']['author']['date'], '%Y-%m-%dT%H:%M:%SZ')
                        commit_history.append({
                            'date': commit_date,
                            'message': commit['commit']['message'],
                            'author': commit['commit']['author']['name'],
                            'sha': commit['sha'][:7]
                        })
                    except:
                        continue
                
                return {
                    'weekly_commits': weekly_commits,
                    'activity_by_week': activity_data,
                    'commit_history': commit_history,
                    'total_commits': len(commits_data)
                }
            elif activity_response.status_code == 403 or commits_response.status_code == 403:
                st.error("Rate limit exceeded. Please try again later.")
            else:
                st.error(f"Failed to fetch commit data. HTTP Status Codes: {activity_response.status_code}, {commits_response.status_code}")
            return None
        except Exception as e:
            st.error(f"Error fetching commit activity: {e}")
            return None

class DataProcessor:
    """Class to process and analyze GitHub data"""
    
    @staticmethod
    def calculate_statistics(activity_data):
        """Calculate repository statistics"""
        weekly_data = activity_data['activity_by_week']
        weekly_totals = [week['total'] for week in weekly_data]
        
        stats = {
            'Total Commits': activity_data['total_commits'],
            'Average Weekly Commits': round(np.mean(weekly_totals), 2),
            'Median Weekly Commits': np.median(weekly_totals),
            'Highest Weekly Activity': max(weekly_totals),
            'Lowest Weekly Activity': min(weekly_totals),
            'Standard Deviation': round(np.std(weekly_totals), 2)
        }
        
        # Calculate activity trend
        recent_weeks = weekly_totals[-12:]  # Last 12 weeks
        trend = "Increasing" if recent_weeks[-1] > np.mean(recent_weeks[:-1]) else "Decreasing"
        stats['Recent Activity Trend'] = trend
        
        return stats
    
    @staticmethod
    def prepare_dataframe_for_export(metadata, contributors, activity_data):
        """Prepare data for CSV export"""
        export_data = []
        
        # Add metadata
        for key, value in metadata.items():
            export_data.append({
                'Category': 'Metadata',
                'Item': key,
                'Value': str(value),
                'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Add contributor data
        for contributor in contributors[:10]:
            export_data.append({
                'Category': 'Contributors',
                'Item': contributor['login'],
                'Value': contributor['contributions'],
                'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Add commit history
        for commit in activity_data['commit_history'][:20]:
            export_data.append({
                'Category': 'Commits',
                'Item': commit['sha'],
                'Value': commit['message'][:100] + '...' if len(commit['message']) > 100 else commit['message'],
                'Date': commit['date'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return pd.DataFrame(export_data)

class Visualizer:
    """Class to handle data visualization"""
    
    @staticmethod
    def create_weekly_activity_chart(activity_data):
        """Create weekly activity chart using Plotly"""
        weekly_data = activity_data['activity_by_week']
        dates = [datetime.utcfromtimestamp(week['week']) for week in weekly_data]
        commits = [week['total'] for week in weekly_data]
        
        fig = px.line(x=dates, y=commits, 
                     title='Weekly Commit Activity (Last 52 Weeks)',
                     labels={'x': 'Date', 'y': 'Number of Commits'})
        fig.update_traces(mode='lines+markers')
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_commit_distribution_chart(activity_data):
        """Create commit distribution histogram"""
        weekly_data = activity_data['activity_by_week']
        commits = [week['total'] for week in weekly_data]
        
        fig = px.histogram(x=commits, nbins=10,
                          title='Distribution of Weekly Commits',
                          labels={'x': 'Commits per Week', 'y': 'Number of Weeks'})
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_contributors_chart(contributors):
        """Create contributors bar chart"""
        top_contributors = contributors[:10]
        names = [c['login'] for c in top_contributors]
        contributions = [c['contributions'] for c in top_contributors]
        
        fig = px.bar(x=names, y=contributions,
                    title='Top 10 Contributors',
                    labels={'x': 'Contributors', 'y': 'Number of Contributions'})
        fig.update_layout(height=400, xaxis_tickangle=-45)
        return fig
    
    @staticmethod
    def create_recent_activity_trend(activity_data):
        """Create recent activity trend chart"""
        weekly_data = activity_data['activity_by_week']
        recent_commits = [week['total'] for week in weekly_data[-12:]]
        weeks = list(range(1, 13))
        
        fig = px.line(x=weeks, y=recent_commits,
                     title='Recent Activity Trend (Last 12 Weeks)',
                     labels={'x': 'Weeks Ago', 'y': 'Number of Commits'})
        fig.update_traces(mode='lines+markers')
        fig.update_layout(height=400)
        return fig

class GitHubRepoAnalyzer:
    """Main analyzer class that orchestrates the analysis"""
    
    def __init__(self):
        self.github_api = GitHubAPI()
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer()
        self.analysis_results = {}
    
    def analyze_repository(self, repo_url):
        """Main method to analyze a GitHub repository"""
        if not repo_url:
            st.error("Please enter a GitHub repository URL")
            return False
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch metadata
        status_text.text("Fetching repository metadata...")
        progress_bar.progress(25)
        metadata = self.github_api.fetch_repo_metadata(repo_url)
        if not metadata:
            return False
        
        # Fetch contributors
        status_text.text("Fetching contributors...")
        progress_bar.progress(50)
        contributors = self.github_api.fetch_contributors(repo_url)
        if not contributors:
            return False
        
        # Fetch commit activity
        status_text.text("Fetching commit activity...")
        progress_bar.progress(75)
        commit_activity = self.github_api.fetch_commit_activity(repo_url)
        if not commit_activity:
            return False
        
        # Store results
        self.analysis_results = {
            'metadata': metadata,
            'contributors': contributors,
            'commit_activity': commit_activity,
            'statistics': self.data_processor.calculate_statistics(commit_activity)
        }
        
        status_text.text("Analysis complete!")
        progress_bar.progress(100)
        
        return True
    
    def display_metadata(self):
        """Display repository metadata"""
        metadata = self.analysis_results['metadata']
        
        st.subheader("📊 Repository Metadata")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("⭐ Stars", metadata['stars'])
            st.metric("🍴 Forks", metadata['forks'])
            st.metric("📊 Size (KB)", metadata['size'])
        
        with col2:
            st.metric("🐛 Open Issues", metadata['open_issues'])
            st.metric("👀 Watchers", metadata['watchers'])
            st.metric("💻 Language", metadata['language'] or "Not specified")
        
        with col3:
            st.metric("📅 Created", metadata['created_at'][:10])
            st.metric("🔄 Updated", metadata['updated_at'][:10])
            st.metric("🌿 Default Branch", metadata['default_branch'])
        
        # Additional information
        st.write(f"**Description:** {metadata['description'] or 'No description available'}")
        st.write(f"**License:** {metadata['license']}")
        
        if metadata['topics']:
            st.write(f"**Topics:** {', '.join(metadata['topics'])}")
    
    def display_contributors(self):
        """Display contributors information"""
        contributors = self.analysis_results['contributors']
        
        st.subheader("👥 Contributors")
        
        # Display contributors chart
        fig = self.visualizer.create_contributors_chart(contributors)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display contributors table
        contributors_df = pd.DataFrame([
            {
                'Login': c['login'],
                'Contributions': c['contributions'],
                'Type': c['type']
            }
            for c in contributors[:15]
        ])
        
        st.dataframe(contributors_df, use_container_width=True)
    
    def display_activity(self):
        """Display commit activity"""
        activity = self.analysis_results['commit_activity']
        
        st.subheader("📈 Commit Activity")
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = self.visualizer.create_weekly_activity_chart(activity)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = self.visualizer.create_commit_distribution_chart(activity)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Recent activity trend
        fig3 = self.visualizer.create_recent_activity_trend(activity)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Recent commits
        st.subheader("📝 Recent Commits")
        recent_commits = activity['commit_history'][:10]
        
        for commit in recent_commits:
            with st.expander(f"{commit['sha']} - {commit['date'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"**Author:** {commit['author']}")
                st.write(f"**Message:** {commit['message']}")
    
    def display_statistics(self):
        """Display repository statistics"""
        stats = self.analysis_results['statistics']
        
        st.subheader("📊 Repository Statistics")
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Commits", stats['Total Commits'])
            st.metric("Average Weekly Commits", stats['Average Weekly Commits'])
        
        with col2:
            st.metric("Median Weekly Commits", stats['Median Weekly Commits'])
            st.metric("Standard Deviation", stats['Standard Deviation'])
        
        with col3:
            st.metric("Highest Weekly Activity", stats['Highest Weekly Activity'])
            st.metric("Recent Trend", stats['Recent Activity Trend'])
    
    def export_analysis(self):
        """Export analysis results"""
        if not self.analysis_results:
            st.error("No analysis results available")
            return
        
        # Prepare data for export
        df = self.data_processor.prepare_dataframe_for_export(
            self.analysis_results['metadata'],
            self.analysis_results['contributors'],
            self.analysis_results['commit_activity']
        )
        
        # Convert to CSV
        csv = df.to_csv(index=False)
        
        # Provide download button
        st.download_button(
            label="📥 Download Analysis Report (CSV)",
            data=csv,
            file_name=f"github_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="GitHub Repository Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 GitHub Repository Analyzer")
    st.write("Analyze GitHub repositories to get insights about metadata, contributors, and commit activity.")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = GitHubRepoAnalyzer()
    
    # Sidebar for input
    with st.sidebar:
        st.header("📋 Configuration")
        repo_url = st.text_input(
            "GitHub Repository URL:",
            placeholder="https://github.com/owner/repository",
            help="Enter the full GitHub repository URL"
        )
        
        analyze_button = st.button("🔍 Analyze Repository", type="primary")
        
        if st.session_state.analyzer.analysis_results:
            st.divider()
            st.session_state.analyzer.export_analysis()
    
    # Main content area
    if analyze_button and repo_url:
        with st.spinner("Analyzing repository..."):
            if st.session_state.analyzer.analyze_repository(repo_url):
                st.success("✅ Analysis completed successfully!")
    
    # Display results if available
    if st.session_state.analyzer.analysis_results:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Metadata", "👥 Contributors", "📈 Activity", "📊 Statistics"])
        
        with tab1:
            st.session_state.analyzer.display_metadata()
        
        with tab2:
            st.session_state.analyzer.display_contributors()
        
        with tab3:
            st.session_state.analyzer.display_activity()
        
        with tab4:
            st.session_state.analyzer.display_statistics()
    
    else:
        # Show welcome message
        st.info("👆 Enter a GitHub repository URL in the sidebar and click 'Analyze Repository' to get started!")
        
        # Example usage
        with st.expander("ℹ️ How to use"):
            st.write("""
            1. **Enter Repository URL**: Paste the GitHub repository URL in the sidebar
            2. **Click Analyze**: Press the 'Analyze Repository' button
            3. **View Results**: Explore the different tabs to see:
               - Repository metadata (stars, forks, issues, etc.)
               - Top contributors and their contributions
               - Commit activity over time
               - Statistical analysis of the repository
            4. **Export Data**: Download the analysis results as a CSV file
            
            **Example URLs:**
            - `https://github.com/microsoft/vscode`
            - `https://github.com/facebook/react`
            - `https://github.com/python/cpython`
            """)

if __name__ == "__main__":
    main()