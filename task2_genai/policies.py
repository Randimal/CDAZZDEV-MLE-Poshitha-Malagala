"""Fictional internal controls, deliberately independent of real laws."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicySpec:
    """Three explicit risk/action rules and two equivalent excerpt styles."""

    topic: str
    category: str
    rules: tuple[tuple[str, str], tuple[str, str], tuple[str, str]]

    def excerpt(self, variant: int = 0) -> str:
        sentences = []
        for risk, (condition, action) in zip(("low", "medium", "high"), self.rules):
            if variant % 2 == 0:
                sentences.append(f"If {condition}, assign {risk} risk and {action}.")
            else:
                sentences.append(
                    f"{risk.capitalize()} risk applies when {condition}; {action}."
                )
        return "Internal policy: " + " ".join(sentences)


POLICIES = (
    PolicySpec(
        "KYC",
        "KYC",
        (
            (
                "identity evidence is complete and consistent",
                "continue identity approval",
            ),
            (
                "identity details have a minor unresolved discrepancy",
                "pause approval and request clarification",
            ),
            (
                "identity evidence is missing or suspected altered",
                "hold onboarding and escalate for verification",
            ),
        ),
    ),
    PolicySpec(
        "AML",
        "AML",
        (
            (
                "activity matches the documented purpose and evidence",
                "continue routine review",
            ),
            (
                "activity is unusual but an explanation awaits verification",
                "request supporting evidence and review",
            ),
            (
                "unexplained layering or concealment indicators remain",
                "hold the activity and escalate to the compliance lead",
            ),
        ),
    ),
    PolicySpec(
        "sanctions",
        "Sanctions",
        (
            (
                "a screening match is disproved by identifiers",
                "document clearance and continue",
            ),
            (
                "a screening match lacks enough identifiers to resolve",
                "pause processing and obtain identifiers",
            ),
            (
                "identifiers confirm an internal restricted-list match",
                "block processing and escalate internally",
            ),
        ),
    ),
    PolicySpec(
        "PEP",
        "PEP",
        (
            (
                "screening confirms no public-role connection",
                "continue standard review",
            ),
            (
                "a possible public-role connection is unverified",
                "pause approval and clarify the connection",
            ),
            (
                "a public role or close connection is confirmed",
                "perform enhanced due diligence and obtain senior approval before onboarding",
            ),
        ),
    ),
    PolicySpec(
        "suspicious transactions",
        "Suspicious Transactions",
        (
            (
                "a flagged transaction has verified commercial purpose",
                "document the explanation and clear the flag",
            ),
            (
                "a flagged transaction has plausible but unverified purpose",
                "request evidence before clearing the flag",
            ),
            (
                "a flagged transaction remains inconsistent with its stated purpose",
                "hold processing and escalate the suspicion",
            ),
        ),
    ),
    PolicySpec(
        "source of funds",
        "Source of Funds",
        (
            (
                "funding origin is supported by consistent documents",
                "accept the source and continue",
            ),
            (
                "funding origin is plausible but documents are incomplete",
                "request missing source evidence",
            ),
            (
                "funding origin is concealed or contradicted by documents",
                "hold acceptance and escalate for enhanced review",
            ),
        ),
    ),
    PolicySpec(
        "enhanced due diligence",
        "Enhanced Due Diligence",
        (
            (
                "enhanced checks and required approval are complete",
                "continue with documented monitoring",
            ),
            (
                "enhanced checks lack a noncritical supporting item",
                "pause approval and obtain the item",
            ),
            (
                "a material concern remains unresolved after enhanced checks",
                "hold approval and escalate to senior compliance",
            ),
        ),
    ),
    PolicySpec(
        "customer due diligence",
        "Customer Due Diligence",
        (
            (
                "ownership and business purpose are verified",
                "approve standard due diligence",
            ),
            (
                "ownership or purpose requires further clarification",
                "request clarification before approval",
            ),
            (
                "ownership is concealed or documents materially conflict",
                "hold approval and escalate for enhanced due diligence",
            ),
        ),
    ),
    PolicySpec(
        "transaction monitoring",
        "Transaction Monitoring",
        (
            ("activity fits the verified profile", "retain routine monitoring"),
            (
                "activity deviates from the profile without verified explanation",
                "review the deviation and request evidence",
            ),
            (
                "repeated unexplained deviations persist after review",
                "hold affected activity and escalate monitoring findings",
            ),
        ),
    ),
    PolicySpec(
        "record keeping",
        "Record Keeping",
        (
            (
                "required review records are complete and retrievable",
                "archive under the internal retention schedule",
            ),
            (
                "a required record is incomplete but recoverable",
                "recover the record and track remediation",
            ),
            (
                "critical evidence is missing or records appear altered",
                "preserve available records and escalate the integrity issue",
            ),
        ),
    ),
    PolicySpec(
        "high-risk jurisdictions",
        "High-Risk Jurisdictions",
        (
            (
                "no internal high-risk location connection is identified",
                "continue standard location review",
            ),
            (
                "a possible internal high-risk location connection is unresolved",
                "clarify the connection before processing",
            ),
            (
                "an internal high-risk location connection is confirmed",
                "perform enhanced review and obtain senior approval",
            ),
        ),
    ),
    PolicySpec(
        "escalation",
        "Escalation",
        (
            (
                "a documented concern is resolved within staff authority",
                "close the concern with evidence",
            ),
            (
                "a concern remains unresolved without immediate material impact",
                "refer it to the compliance reviewer",
            ),
            (
                "a material unresolved concern requires restricted authority",
                "hold the affected decision and escalate to the compliance lead",
            ),
        ),
    ),
    PolicySpec(
        "onboarding controls",
        "Onboarding Controls",
        (
            (
                "all required onboarding controls are complete",
                "activate the account after approval",
            ),
            (
                "a required noncritical onboarding check is pending",
                "keep activation paused until completion",
            ),
            (
                "a critical check fails or approval is bypassed",
                "block activation and escalate the control breach",
            ),
        ),
    ),
    PolicySpec(
        "false positives",
        "False Positives",
        (
            (
                "independent identifiers disprove an alert",
                "document the mismatch and clear the alert",
            ),
            (
                "an alert is plausible but identifiers are inconclusive",
                "retain the alert and obtain distinguishing evidence",
            ),
            (
                "independent identifiers substantiate an alert",
                "retain the restriction and escalate the confirmed concern",
            ),
        ),
    ),
)
POLICY_BY_TOPIC = {policy.topic: policy for policy in POLICIES}
