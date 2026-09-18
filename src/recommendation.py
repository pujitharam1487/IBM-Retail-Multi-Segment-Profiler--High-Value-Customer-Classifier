"""
Recommendation Engine Module.
Generates tailored, data-driven marketing strategies, promotional policies,
and campaign tactics based on customer persona and high-value probability.
"""

from typing import Dict, Any, List


class RetailRecommendationEngine:
    """
    Translates customer persona attributes and high-value predictions into actionable marketing strategies.
    """

    @staticmethod
    def get_recommendation(
        persona_name: str,
        is_high_value: int,
        high_value_prob: float,
        annual_income: float,
        spending_score: float
    ) -> Dict[str, Any]:
        """
        Computes a comprehensive marketing recommendation package.
        """
        # Standardize persona matching
        p_name = persona_name.lower()
        
        if "vip" in p_name or "premium" in p_name or (annual_income >= 70 and spending_score >= 65):
            strategy = "VIP Loyalty Concierge & Exclusive Retention"
            badge = "💎 VIP Elite"
            channel = "Dedicated Concierge / Private Invitation / SMS Preview"
            tactics = [
                "Invite to exclusive private previews of new product collections before public release.",
                "Assign personal styling consultant and priority customer support hotline.",
                "Offer complimentary luxury packaging, free expedited shipping, and anniversary gifts.",
                "Provide invitations to VIP closed-door gala events and designer masterclasses."
            ]
            promo_policy = "Zero price discounting. Retain brand prestige with experiential and value-added luxury perks."
            expected_roi = "+22% Customer Lifetime Value (CLV), 95%+ annual retention rate."

        elif "saver" in p_name or "conservative" in p_name or (annual_income >= 70 and spending_score < 65):
            strategy = "High-Value Upselling & Premium Basket Conversion"
            badge = "🎯 High Potential Whales"
            channel = "Curated Email Lookbooks / Targeted Category Highlights"
            tactics = [
                "Target with high-ticket bundle offers (e.g., 'Spend $300, receive premium travel kit').",
                "Highlight product durability, craftsmanship, quality assurance, and extended warranties.",
                "Personalized product recommendations based on high-margin product categories.",
                "Limited-time upgrades to higher loyalty reward tier upon next purchase."
            ]
            promo_policy = "Tiered threshold incentives ($50 voucher for orders above $250) rather than generic discounts."
            expected_roi = "+35% Average Order Value (AOV), converting latent wealth into active revenue."

        elif "trendsetter" in p_name or "enthusiast" in p_name or (annual_income < 70 and spending_score >= 65):
            strategy = "Gamified Mobile Engagement & Flash Social Drops"
            badge = "🔥 Trendsetters"
            channel = "Mobile App Push / TikTok & Instagram Ads / SMS Alerts"
            tactics = [
                "Launch 24-hour flash drops and limited-edition product releases via mobile app.",
                "Incentivize viral social sharing with dual-sided referral bonuses ($15 for both parties).",
                "Gamify shopping experience with streak badges, reviews rewards, and app points.",
                "Provide flexible 'Buy Now, Pay Later' (BNPL) payment integrations at checkout."
            ]
            promo_policy = "Time-sensitive flash promotions and app-exclusive coupon codes."
            expected_roi = "+40% purchase frequency, high viral acquisition of peer demographics."

        elif "budget" in p_name or "frugal" in p_name or (annual_income < 70 and spending_score < 40):
            strategy = "Smart Value & Price-Sensitive Promotional Campaigns"
            badge = "🏷️ Value Seekers"
            channel = "Weekly Promotional Email / Discount Catalog SMS"
            tactics = [
                "Highlight seasonal clearance sales, outlet promotions, and bundled value packs.",
                "Implement 'Buy 2 Get 1 Free' or multi-pack savings offers on everyday staples.",
                "Promote price-match guarantees and transparent value comparisons.",
                "Send cart-abandonment notifications with incremental 10% discount codes."
            ]
            promo_policy = "Volume and clearance discounts to maximize inventory turnover."
            expected_roi = "+15% conversion on discounted inventory, clearing off-season stock."

        else:
            strategy = "Mainstream Engagement & Cross-Category Discovery"
            badge = "⚖️ Core Shoppers"
            channel = "Monthly Digest / In-App Rewards Tracker / Email"
            tactics = [
                "Encourage category expansion with 'You Might Also Like' personalized recommendations.",
                "Reward ongoing patronage with points-multiplier weekends (e.g. 2x points on Saturdays).",
                "Send personalized birthday month bonus credits.",
                "Re-engage at 45-day intervals with curated product showcases."
            ]
            promo_policy = "Loyalty points bonuses and moderate promotional discounts."
            expected_roi = "+18% cross-sell basket penetration and steady customer retention."

        # High-Value modifier
        if is_high_value == 1 and high_value_prob >= 0.70:
            status_label = "Confirmed High-Value Asset"
            priority = "Priority 1 - Immediate Account Cultivation"
        elif high_value_prob >= 0.50:
            status_label = "Rising High-Value Potential"
            priority = "Priority 2 - Nurture & Basket Escalation"
        else:
            status_label = "Standard Retail Customer"
            priority = "Priority 3 - Standard Automated Engagement"

        return {
            "strategy": strategy,
            "badge": badge,
            "channel": channel,
            "tactics": tactics,
            "promo_policy": promo_policy,
            "expected_roi": expected_roi,
            "status_label": status_label,
            "priority": priority,
            "high_value_probability": round(high_value_prob, 3),
            "is_high_value": is_high_value
        }
