"""
StemScribe AI Agent Teams
=========================
Two teams of AI agents powered by CrewAI:
1. Research Team - Market analysis, competitor tracking, user insights
2. Marketing Team - Content creation, social media, SEO, email, ads

Usage:
    python agents.py --team research --task competitor_report
    python agents.py --team marketing --task weekly_content
    python agents.py --team both --task full_pipeline
"""

import os
import yaml
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    FileReadTool,
)
from dotenv import load_dotenv

load_dotenv()

# Load config
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


# =============================================================================
# RESEARCH TEAM AGENTS
# =============================================================================

class ResearchTeam:
    """AI Research team for market intelligence and product strategy."""

    def __init__(self, llm_model: str = "anthropic/claude-sonnet-4-5-20250929"):
        self.llm = llm_model
        self.web_search = SerperDevTool()
        self.web_scrape = ScrapeWebsiteTool()
        self.file_reader = FileReadTool()

    def market_research_agent(self) -> Agent:
        return Agent(
            role="Market Research Analyst",
            goal=(
                "Analyze the music technology market to identify trends, "
                "opportunities, and threats for StemScribe. Track competitor "
                "movements and pricing changes."
            ),
            backstory=(
                "You are a senior market research analyst specializing in "
                "music technology and SaaS products. You have deep knowledge "
                "of the stem separation, music transcription, and music "
                "education markets. You track Moises, Chordify, LALAL.AI, "
                "and other competitors obsessively."
            ),
            tools=[self.web_search, self.web_scrape],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )

    def user_research_agent(self) -> Agent:
        return Agent(
            role="User Research Specialist",
            goal=(
                "Understand what musicians need from stem separation and "
                "transcription tools. Monitor Reddit, forums, and social "
                "media for pain points, feature requests, and sentiment."
            ),
            backstory=(
                "You are a user research specialist who is also a musician. "
                "You understand the needs of jazz players, worship teams, "
                "music students, producers, and hobbyists. You monitor "
                "r/musictheory, r/WeAreTheMusicMakers, r/Guitar, "
                "r/jazz, Gearslutz, TalkBass, and other music communities."
            ),
            tools=[self.web_search, self.web_scrape],
            llm=self.llm,
            verbose=True,
        )

    def pricing_strategy_agent(self) -> Agent:
        return Agent(
            role="Pricing & Revenue Strategist",
            goal=(
                "Optimize StemScribe's pricing model for maximum revenue "
                "and user acquisition. Analyze competitor pricing, identify "
                "willingness-to-pay segments, and recommend pricing experiments."
            ),
            backstory=(
                "You are a SaaS pricing expert who has helped multiple "
                "music and creative tool companies optimize their monetization. "
                "You understand freemium conversion, subscription fatigue, "
                "and credit-based pricing models."
            ),
            tools=[self.web_search, self.file_reader],
            llm=self.llm,
            verbose=True,
        )

    # --- RESEARCH TASKS ---

    def competitor_analysis_task(self) -> Task:
        return Task(
            description=(
                "Conduct a comprehensive competitor analysis for StemScribe. "
                "Research the following competitors: Moises, Chordify, LALAL.AI, "
                "RipX, iZotope RX, and any new entrants. For each, identify:\n"
                "1. Current pricing and any recent changes\n"
                "2. New features launched in the last 3 months\n"
                "3. User reviews and common complaints\n"
                "4. App store ratings and download trends\n"
                "5. Marketing strategies they're using\n"
                "6. Gaps that StemScribe could exploit\n\n"
                "Output a structured markdown report."
            ),
            expected_output="A detailed competitor analysis report in markdown format.",
            agent=self.market_research_agent(),
            output_file="reports/competitor_analysis.md",
        )

    def user_pain_points_task(self) -> Task:
        return Task(
            description=(
                "Search Reddit (r/musictheory, r/WeAreTheMusicMakers, r/Guitar, "
                "r/jazz, r/Bass, r/bluegrass, r/gratefulDead), music forums, "
                "and social media for:\n"
                "1. Common complaints about existing stem separation tools\n"
                "2. Feature requests that current tools don't fulfill\n"
                "3. Price sensitivity discussions\n"
                "4. Use cases people describe (learning songs, creating backing "
                "tracks, remixing, worship, etc.)\n"
                "5. Specific genres underserved by current tools\n\n"
                "Compile into a user insights report with direct quotes and links."
            ),
            expected_output="A user insights report with categorized pain points and opportunities.",
            agent=self.user_research_agent(),
            output_file="reports/user_insights.md",
        )

    def pricing_optimization_task(self) -> Task:
        return Task(
            description=(
                f"Analyze StemScribe's proposed pricing:\n"
                f"- Free: {CONFIG['pricing_tiers']['free']['features']}\n"
                f"- Premium: ${CONFIG['pricing_tiers']['premium']['price']}/mo\n"
                f"- Pro: ${CONFIG['pricing_tiers']['pro']['price']}/mo\n\n"
                "Compare against competitor pricing:\n"
                f"- Moises: {CONFIG['competitors'][0]['pricing']}\n"
                f"- Chordify: {CONFIG['competitors'][1]['pricing']}\n"
                f"- LALAL.AI: {CONFIG['competitors'][2]['pricing']}\n\n"
                "Recommend:\n"
                "1. Is the pricing competitive?\n"
                "2. Should we use subscription vs credits vs hybrid?\n"
                "3. What features should gate each tier?\n"
                "4. Annual discount strategy\n"
                "5. Student/educator discount program\n"
                "6. Launch promotion strategy"
            ),
            expected_output="A pricing strategy recommendation document.",
            agent=self.pricing_strategy_agent(),
            output_file="reports/pricing_strategy.md",
        )

    def build_crew(self) -> Crew:
        return Crew(
            agents=[
                self.market_research_agent(),
                self.user_research_agent(),
                self.pricing_strategy_agent(),
            ],
            tasks=[
                self.competitor_analysis_task(),
                self.user_pain_points_task(),
                self.pricing_optimization_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )


# =============================================================================
# MARKETING TEAM AGENTS
# =============================================================================

class MarketingTeam:
    """AI Marketing team for content, social, SEO, and email."""

    def __init__(self, llm_model: str = "anthropic/claude-sonnet-4-5-20250929"):
        self.llm = llm_model
        self.web_search = SerperDevTool()
        self.web_scrape = ScrapeWebsiteTool()
        self.file_reader = FileReadTool()

    def content_strategist_agent(self) -> Agent:
        return Agent(
            role="Content Strategist & Writer",
            goal=(
                "Create compelling blog posts, tutorials, and educational "
                "content that drives organic traffic and positions StemScribe "
                "as the go-to tool for musicians who want to learn songs, "
                "separate stems, and analyze chord progressions."
            ),
            backstory=(
                "You are a content marketing expert who is also a musician. "
                "You write engaging, SEO-optimized content about music "
                "technology, music theory, and music education. You know "
                "how to write for jazz players, worship teams, producers, "
                "and bedroom guitarists. Your content converts readers into "
                "users."
            ),
            tools=[self.web_search, self.file_reader],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )

    def social_media_agent(self) -> Agent:
        return Agent(
            role="Social Media Manager",
            goal=(
                "Grow StemScribe's social media presence across YouTube, "
                "TikTok, Reddit, Instagram, and Twitter. Create engaging "
                "posts that showcase stem separation demos, chord recognition "
                "features, and musician stories."
            ),
            backstory=(
                "You are a social media manager who specializes in music "
                "and creative tool brands. You know what goes viral in "
                "music communities - before/after stem separation clips, "
                "isolated instrument reveals, chord breakdown videos. "
                "You understand each platform's algorithm and culture."
            ),
            tools=[self.web_search],
            llm=self.llm,
            verbose=True,
        )

    def seo_specialist_agent(self) -> Agent:
        return Agent(
            role="SEO Specialist",
            goal=(
                "Optimize StemScribe's web presence for organic search. "
                "Target high-intent keywords like 'stem separation', "
                "'AI music transcription', and 'chord recognition app'. "
                "Build topical authority in music technology."
            ),
            backstory=(
                "You are an SEO expert specializing in SaaS and music "
                "technology. You understand keyword research, on-page "
                "optimization, content clustering, and link building "
                "strategies specific to the music tools niche."
            ),
            tools=[self.web_search, self.web_scrape],
            llm=self.llm,
            verbose=True,
        )

    def email_marketing_agent(self) -> Agent:
        return Agent(
            role="Email Marketing Specialist",
            goal=(
                "Build and optimize email sequences that convert free "
                "users to paid subscribers. Create welcome sequences, "
                "feature highlights, and retention campaigns."
            ),
            backstory=(
                "You are an email marketing specialist for SaaS products. "
                "You write compelling subject lines, design drip campaigns, "
                "and optimize for open rates and conversions. You know "
                "that musicians are busy and need concise, valuable emails."
            ),
            tools=[self.file_reader],
            llm=self.llm,
            verbose=True,
        )

    def community_manager_agent(self) -> Agent:
        return Agent(
            role="Community Manager & Growth Hacker",
            goal=(
                "Build grassroots community presence in music forums, "
                "Reddit, Discord, and Facebook groups. Identify and engage "
                "with potential users authentically. Find partnership and "
                "collaboration opportunities."
            ),
            backstory=(
                "You are a community manager who is genuinely passionate "
                "about music. You know the culture of music communities "
                "and never come across as spammy or salesy. You provide "
                "value first - answering questions, sharing tips - and "
                "mention StemScribe only when genuinely relevant."
            ),
            tools=[self.web_search, self.web_scrape],
            llm=self.llm,
            verbose=True,
        )

    # --- MARKETING TASKS ---

    def weekly_blog_post_task(self) -> Task:
        keywords = CONFIG["seo_keywords"]["long_tail"]
        return Task(
            description=(
                "Write an SEO-optimized blog post for StemScribe's website. "
                "Choose one of these target keywords and write a 1200-1500 "
                "word article:\n"
                f"Keywords: {keywords}\n\n"
                "Requirements:\n"
                "1. Engaging title with the primary keyword\n"
                "2. Meta description (155 chars max)\n"
                "3. Introduction that hooks the reader\n"
                "4. 3-5 H2 sections with clear structure\n"
                "5. Naturally mention StemScribe as a solution (not salesy)\n"
                "6. Include a CTA to try StemScribe free\n"
                "7. Internal links to other blog posts (suggest placeholders)\n"
                "8. Output as markdown with frontmatter"
            ),
            expected_output="A complete blog post in markdown with frontmatter.",
            agent=self.content_strategist_agent(),
            output_file="content/blog_post.md",
        )

    def social_media_calendar_task(self) -> Task:
        return Task(
            description=(
                "Create a 2-week social media content calendar for StemScribe. "
                "Include posts for:\n"
                "- YouTube (2 videos): Demo videos showing stem separation\n"
                "- TikTok/Reels (5 posts): Short viral clips\n"
                "- Reddit (3 posts): Valuable discussions in music subreddits\n"
                "- Twitter (5 tweets): Engagement + announcements\n\n"
                "For each post, include:\n"
                "1. Platform\n"
                "2. Post date/time (optimize for musician audience)\n"
                "3. Content description / script\n"
                "4. Hashtags (where applicable)\n"
                "5. CTA\n"
                "6. Expected engagement type (views, clicks, signups)\n\n"
                "Format as a structured markdown table."
            ),
            expected_output="A 2-week social media calendar in markdown table format.",
            agent=self.social_media_agent(),
            output_file="content/social_calendar.md",
        )

    def seo_audit_task(self) -> Task:
        return Task(
            description=(
                "Create an SEO content strategy for StemScribe targeting "
                "these keyword clusters:\n\n"
                f"Primary keywords: {CONFIG['seo_keywords']['primary']}\n"
                f"Secondary keywords: {CONFIG['seo_keywords']['secondary']}\n"
                f"Long-tail keywords: {CONFIG['seo_keywords']['long_tail']}\n\n"
                "Deliverables:\n"
                "1. Content cluster map (pillar pages + supporting articles)\n"
                "2. Keyword difficulty estimates (low/medium/high)\n"
                "3. 10 blog post titles targeting specific keywords\n"
                "4. On-page SEO checklist for the main landing page\n"
                "5. Competitor content gaps (what they rank for that we don't)\n"
                "6. Link building opportunities in the music niche"
            ),
            expected_output="An SEO strategy document with content clusters and keyword targets.",
            agent=self.seo_specialist_agent(),
            output_file="content/seo_strategy.md",
        )

    def email_sequence_task(self) -> Task:
        return Task(
            description=(
                "Write the complete welcome email sequence for new StemScribe "
                "users (free tier signups). Create 5 emails:\n\n"
                "Email 1 (Day 0): Welcome + quick start guide\n"
                "Email 2 (Day 2): Tutorial - separate your first song\n"
                "Email 3 (Day 5): Feature highlight - chord recognition\n"
                "Email 4 (Day 10): Social proof + Premium upsell\n"
                "Email 5 (Day 14): Last chance discount offer\n\n"
                "For each email:\n"
                "1. Subject line (+ 2 A/B test variants)\n"
                "2. Preview text\n"
                "3. Body copy (conversational, musician-friendly tone)\n"
                "4. CTA button text\n"
                "5. Segment targeting notes\n\n"
                "Keep emails short (under 200 words each). Musicians are busy."
            ),
            expected_output="5 complete welcome emails with subject lines and body copy.",
            agent=self.email_marketing_agent(),
            output_file="content/email_welcome_sequence.md",
        )

    def community_outreach_task(self) -> Task:
        return Task(
            description=(
                "Create a community outreach plan for StemScribe. Include:\n\n"
                "1. Reddit strategy:\n"
                "   - Which subreddits to engage in (with subscriber counts)\n"
                "   - Sample helpful comments (provide value, not spam)\n"
                "   - Thread ideas to post that would naturally showcase StemScribe\n\n"
                "2. Partnership opportunities:\n"
                "   - Music education YouTube channels to collaborate with\n"
                "   - Music blogs that review tools\n"
                "   - Podcast guest opportunities\n"
                "   - Music teacher/professor partnerships\n\n"
                "3. Discord/Facebook groups:\n"
                "   - Relevant music communities to join\n"
                "   - Engagement strategy (give first, promote later)\n\n"
                "4. Product Hunt launch plan:\n"
                "   - Timing, assets needed, launch day strategy\n"
                "   - Hunter outreach template"
            ),
            expected_output="A community outreach and partnership plan.",
            agent=self.community_manager_agent(),
            output_file="content/community_plan.md",
        )

    def build_crew(self) -> Crew:
        return Crew(
            agents=[
                self.content_strategist_agent(),
                self.social_media_agent(),
                self.seo_specialist_agent(),
                self.email_marketing_agent(),
                self.community_manager_agent(),
            ],
            tasks=[
                self.weekly_blog_post_task(),
                self.social_media_calendar_task(),
                self.seo_audit_task(),
                self.email_sequence_task(),
                self.community_outreach_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )


# =============================================================================
# CLI RUNNER
# =============================================================================

if __name__ == "__main__":
    import argparse
    from rich.console import Console

    console = Console()

    parser = argparse.ArgumentParser(description="StemScribe AI Agent Teams")
    parser.add_argument(
        "--team",
        choices=["research", "marketing", "both"],
        default="both",
        help="Which team to run",
    )
    parser.add_argument(
        "--task",
        default="full_pipeline",
        help="Specific task or 'full_pipeline' for all",
    )
    args = parser.parse_args()

    # Create output directories
    os.makedirs("reports", exist_ok=True)
    os.makedirs("content", exist_ok=True)

    if args.team in ("research", "both"):
        console.print("\n[bold green]>>> LAUNCHING RESEARCH TEAM <<<[/bold green]\n")
        research = ResearchTeam()
        research_crew = research.build_crew()
        research_result = research_crew.kickoff()
        console.print("\n[bold green]>>> RESEARCH COMPLETE <<<[/bold green]")
        console.print(research_result)

    if args.team in ("marketing", "both"):
        console.print("\n[bold blue]>>> LAUNCHING MARKETING TEAM <<<[/bold blue]\n")
        marketing = MarketingTeam()
        marketing_crew = marketing.build_crew()
        marketing_result = marketing_crew.kickoff()
        console.print("\n[bold blue]>>> MARKETING COMPLETE <<<[/bold blue]")
        console.print(marketing_result)
