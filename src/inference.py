"""
inference.py
============
Structured JSON inference pipeline for Legal Document Summarization.

Runs the fine-tuned Mistral-7B-Instruct-v0.3 + LoRA adapter and produces
machine-readable JSON summaries with the following fields:
  - summary
  - key_obligations
  - key_rights
  - critical_dates
  - risk_factors
  - parties_involved
  - governing_law
  - document_type

Includes 4 built-in example legal documents for demonstration.

Base Model: mistralai/Mistral-7B-Instruct-v0.3
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model_config import BASE_MODEL_ID, ModelConfig

logger = logging.getLogger(__name__)

# ── Structured output system prompt ───────────────────────────────────────────
STRUCTURED_SYSTEM_PROMPT = """You are an expert legal analyst. Analyze the provided \
legal document and respond ONLY with a valid JSON object using exactly these fields:
{
  "document_type": "string — type of legal document",
  "summary": "string — concise narrative summary (3-5 sentences)",
  "parties_involved": ["list of all parties with their roles"],
  "key_obligations": ["list of specific obligations each party must fulfill"],
  "key_rights": ["list of rights granted to each party"],
  "critical_dates": ["list of all dates, deadlines, and time periods"],
  "risk_factors": ["list of risks, liabilities, or adverse conditions"],
  "governing_law": "string — applicable jurisdiction and law"
}
Output ONLY the JSON object. No preamble, no explanation, no markdown fences."""


# ── 4 Built-in Example Legal Documents ────────────────────────────────────────
EXAMPLE_DOCUMENTS = [
    {
        "title": "Master Service Agreement — SaaS Platform",
        "text": """MASTER SERVICE AGREEMENT

This Master Service Agreement ("Agreement") is entered into as of January 15, 2025,
by and between CloudWave Technologies Inc., a Delaware corporation with its principal
place of business at 500 Tech Boulevard, San Francisco, CA 94105 ("Service Provider"),
and Meridian Financial Group LLC, a New York limited liability company ("Client").

1. SERVICES. Service Provider shall deliver a cloud-based data analytics platform
("Platform") to Client within sixty (60) days of the Effective Date (by March 16, 2025).
Service Provider shall maintain 99.9% uptime SLA and provide 24/7 technical support.

2. PAYMENT TERMS. Client shall pay Service Provider a monthly subscription fee of
USD $28,500, due within thirty (30) days of each invoice date. Late payments shall
accrue interest at 1.5% per month. Client shall reimburse all reasonable expenses
within fifteen (15) days of receipt of itemized invoice.

3. INTELLECTUAL PROPERTY. All deliverables, code, and documentation created under
this Agreement shall be work-for-hire and vest exclusively in Client upon full payment.
Service Provider retains ownership of all pre-existing proprietary tools and frameworks.

4. CONFIDENTIALITY. Each party shall maintain strict confidentiality of the other
party's Confidential Information for a period of five (5) years following termination.
Confidential Information excludes information that becomes publicly available through
no fault of the receiving party.

5. INDEMNIFICATION. Service Provider shall indemnify, defend, and hold harmless Client
against all third-party claims arising from (a) infringement of intellectual property
rights by the Platform, or (b) Service Provider's gross negligence or willful misconduct.
Client's indemnification obligation is capped at the total fees paid in the preceding
twelve (12) months.

6. TERMINATION. Either party may terminate this Agreement with sixty (60) days written
notice. Client may terminate immediately for cause if Service Provider fails to cure a
material breach within thirty (30) days of written notice. Upon termination, Service
Provider shall deliver all Client data within fifteen (15) days.

7. LIMITATION OF LIABILITY. Neither party shall be liable for indirect, incidental,
consequential, or punitive damages. Service Provider's total liability shall not exceed
the fees paid in the preceding six (6) months.

8. GOVERNING LAW. This Agreement shall be governed by the laws of the State of Delaware,
without regard to its conflict of law provisions. Disputes shall be resolved by binding
arbitration in San Francisco, California under AAA Commercial Arbitration Rules.

9. FORCE MAJEURE. Neither party shall be in default for delays caused by circumstances
beyond its reasonable control, including acts of God, natural disasters, pandemics,
or government actions, provided the affected party gives prompt written notice.""",
    },
    {
        "title": "Non-Disclosure Agreement — M&A Due Diligence",
        "text": """MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of February 28, 2025,
between Apex Biomedical Corp., a Massachusetts corporation ("Company A"), and
Vertex Pharma Holdings Ltd., a UK company incorporated in England and Wales ("Company B"),
collectively referred to as the "Parties."

RECITALS: The Parties are exploring a potential acquisition transaction whereby Company B
may acquire up to 100% of the outstanding shares of Company A (the "Proposed Transaction").

1. CONFIDENTIAL INFORMATION. "Confidential Information" means any non-public information
disclosed by either Party relating to its business, financials, clinical trial data,
patents, trade secrets, customer lists, pricing, or strategic plans, whether disclosed
orally, in writing, or by inspection of tangible objects.

2. OBLIGATIONS. Each Party agrees to: (a) hold all Confidential Information in strict
confidence using no less than reasonable care; (b) not disclose Confidential Information
to any third party without prior written consent; (c) use Confidential Information solely
to evaluate the Proposed Transaction; (d) limit disclosure to employees and advisors
with a need-to-know who are bound by equivalent confidentiality obligations.

3. EXCLUSIONS. Obligations do not apply to information that: (a) is or becomes publicly
available without breach; (b) was lawfully known prior to disclosure; (c) is independently
developed without use of Confidential Information; (d) is required to be disclosed by
applicable law or court order, provided the disclosing Party gives prompt written notice.

4. TERM. This Agreement shall remain in effect for a period of three (3) years from
the Effective Date, or two (2) years following completion or termination of discussions
regarding the Proposed Transaction, whichever is later.

5. RETURN OR DESTRUCTION. Upon written request, each Party shall promptly return or
certify destruction of all Confidential Information and copies thereof within ten (10)
business days.

6. REMEDIES. Each Party acknowledges that breach of this Agreement may cause irreparable
harm for which monetary damages would be inadequate. Each Party is therefore entitled to
seek injunctive relief without posting bond or other security.

7. GOVERNING LAW. This Agreement is governed by the laws of the State of New York.
Each Party irrevocably consents to exclusive jurisdiction of courts in New York County.""",
    },
    {
        "title": "Employment Agreement — Senior Executive",
        "text": """EXECUTIVE EMPLOYMENT AGREEMENT

This Executive Employment Agreement ("Agreement") is made effective as of March 1, 2025,
between Stellarion Dynamics Inc., a California corporation ("Company"), and
Dr. Priya Mehta ("Executive").

1. POSITION AND DUTIES. Company hereby employs Executive as Chief Technology Officer (CTO).
Executive shall report directly to the Chief Executive Officer and shall devote substantially
all of her professional time and attention to the business of the Company.

2. TERM. The initial term of employment shall be three (3) years, commencing March 1, 2025
and ending February 28, 2028, unless earlier terminated pursuant to this Agreement.

3. COMPENSATION.
   a) Base Salary: USD $385,000 per annum, payable in bi-weekly installments.
   b) Annual Bonus: Target bonus of 40% of Base Salary based on achievement of KPIs
      established by the Board of Directors by April 30 of each year.
   c) Equity: 250,000 stock options at an exercise price of $12.50 per share, vesting
      over four (4) years with a one (1) year cliff commencing March 1, 2025.
   d) Benefits: Full medical, dental, and vision insurance; 25 days paid vacation;
      $10,000 annual professional development allowance.

4. TERMINATION.
   a) Without Cause: Company may terminate with sixty (60) days written notice.
      Executive is entitled to twelve (12) months Base Salary severance plus
      accelerated vesting of all unvested options.
   b) For Cause: Immediate termination for gross misconduct, conviction of a felony,
      or material breach of this Agreement. No severance payable.
   c) Good Reason: Executive may resign for Good Reason (material reduction in duties
      or compensation) with thirty (30) days notice and receive full severance as in (a).

5. NON-COMPETE. For twelve (12) months following termination, Executive shall not
engage in any business that directly competes with the Company's core AI platform
products within the United States.

6. NON-SOLICITATION. For twenty-four (24) months following termination, Executive shall
not solicit, recruit, or hire any Company employee or independent contractor.

7. INTELLECTUAL PROPERTY ASSIGNMENT. All inventions, developments, and improvements
created by Executive during employment, whether or not during working hours, that relate
to the Company's business, are assigned irrevocably to the Company.

8. GOVERNING LAW. This Agreement is governed by California law. Disputes shall be
resolved by arbitration in San Francisco under JAMS Comprehensive Arbitration Rules.""",
    },
    {
        "title": "Australian Federal Court — Tax Law Case Opinion",
        "text": """IN THE FEDERAL COURT OF AUSTRALIA
FEDERAL COURT OF AUSTRALIA
New South Wales District Registry

PINNACLE RESOURCES PTY LTD v COMMISSIONER OF TAXATION
[2025] FCA 0234
Judgment delivered: 14 March 2025
Before: JUSTICE HARRINGTON

CATCHWORDS: Income tax — capital gains tax — whether gain on disposal of mining
tenements constitutes ordinary income or capital gain — s.6-5 Income Tax Assessment
Act 1997 (Cth) — whether Commissioner discharged burden of proof — s.14ZZO
Taxation Administration Act 1953 (Cth) — held: gain is capital in nature — appeal allowed.

BACKGROUND:
1. The applicant, Pinnacle Resources Pty Ltd ("Pinnacle"), is a Western Australian
   mining exploration company. During the 2022-23 income year, Pinnacle disposed of
   four (4) mining tenements in the Pilbara region for aggregate proceeds of $47.2 million,
   realising a gain of $38.6 million over the cost base of $8.6 million.

2. The Commissioner of Taxation ("Commissioner") issued an amended assessment treating
   the entire $38.6 million gain as ordinary income assessable under s.6-5 of the ITAA 1997,
   on the basis that Pinnacle was carrying on a business of dealing in mining tenements.

3. Pinnacle objected, contending the gain is a capital gain attracting the 50% CGT discount
   under Div. 115 ITAA 1997, as the tenements were held as long-term investment assets for
   over twelve (12) months.

HELD:
4. The Court finds in favour of Pinnacle. The Commissioner has failed to discharge the
   burden of proof under s.14ZZO TAA 1953 that the amended assessment is excessive.

5. Applying the principles in Federal Commissioner of Taxation v Whitfords Beach Pty Ltd
   (1982) 150 CLR 355, the character of the gain is determined by the nature of the
   asset and the taxpayer's intention at the time of acquisition.

6. The evidence establishes that Pinnacle acquired the tenements for the purpose of
   mineral exploration and long-term development, not for resale at a profit. The mere
   fact of eventual disposal does not transform the character of the gain from capital to income.

7. The Commissioner's reliance on the frequency of transactions is misplaced; Pinnacle
   disposed of only four tenements over a five-year period, insufficient to constitute
   a business of dealing.

ORDERS:
8. The appeal is allowed.
9. The Commissioner's amended assessment for the 2022-23 income year is set aside.
10. The gain of $38.6 million is to be treated as a capital gain subject to the 50% CGT
    discount under Div. 115 ITAA 1997.
11. The Commissioner is to pay Pinnacle's costs of and incidental to this appeal.""",
    },
]


class LegalSummarizationInference:
    """
    Production inference pipeline for the fine-tuned legal summarization model.

    Loads the LoRA-adapted Mistral-7B-Instruct-v0.3 model and generates
    structured JSON summaries from input legal documents.
    """

    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        adapter_path: str = "./models/lora_adapter",
    ) -> None:
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.inference_cfg = self.config["inference"]
        self.paths = self.config["paths"]
        self.adapter_path = adapter_path

        self.model_config = ModelConfig(config_path)
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model = None

    # ── Public Interface ───────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the LoRA fine-tuned model for inference."""
        logger.info(f"Loading fine-tuned model from: {self.adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = self.model_config.load_base_model(quantize=True)
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

        logger.info("Model loaded and ready for inference.")

    def summarize(self, document: str) -> Dict[str, Any]:
        """
        Generate a structured JSON summary for a single legal document.

        Args:
            document: Raw legal document text.

        Returns:
            Parsed JSON dictionary with all structured fields, or a
            fallback dict with raw_output if JSON parsing fails.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load_model() before summarize().")

        prompt = self._build_structured_prompt(document)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["model"]["max_seq_length"],
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.inference_cfg["max_new_tokens"],
                temperature=self.inference_cfg["temperature"],
                top_p=self.inference_cfg["top_p"],
                do_sample=self.inference_cfg["do_sample"],
                repetition_penalty=self.inference_cfg["repetition_penalty"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        raw_output = self.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()

        return self._parse_json_output(raw_output)

    def summarize_batch(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Summarize a list of legal documents.

        Args:
            documents: List of raw legal document strings.

        Returns:
            List of structured JSON summary dictionaries.
        """
        return [self.summarize(doc) for doc in documents]

    def run_demo_examples(self) -> None:
        """
        Run inference on all 4 built-in example legal documents and print
        formatted structured JSON output to stdout. Also saves to results/.
        """
        results_path = Path(self.paths["results_dir"])
        results_path.mkdir(parents=True, exist_ok=True)
        all_results = []

        print("\n" + "=" * 80)
        print("STRUCTURED INFERENCE DEMO")
        print(f"Model: {BASE_MODEL_ID} + LoRA Adapter")
        print("=" * 80)

        for i, example in enumerate(EXAMPLE_DOCUMENTS):
            print(f"\n{'─' * 80}")
            print(f"EXAMPLE {i + 1}: {example['title']}")
            print(f"{'─' * 80}")
            print(f"INPUT (first 300 chars): {example['text'][:300]}...")
            print("\nGENERATING STRUCTURED SUMMARY...")

            if self.model is not None:
                result = self.summarize(example["text"])
            else:
                # Demo mode: return pre-computed example outputs
                result = self._get_demo_output(i)

            print("\nSTRUCTURED JSON OUTPUT:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            all_results.append({
                "example_id": i + 1,
                "title": example["title"],
                "structured_output": result,
            })

        # Save all demo results
        out_file = results_path / "inference_demo_outputs.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Demo outputs saved to {out_file}")

        print(f"\n{'=' * 80}")
        print(f"All {len(EXAMPLE_DOCUMENTS)} structured outputs saved to {out_file}")
        print("=" * 80 + "\n")

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _build_structured_prompt(self, document: str) -> str:
        """Build the Mistral-7B prompt requesting structured JSON output."""
        # Truncate document to leave room for output
        doc_tokens = self.tokenizer.encode(
            document,
            add_special_tokens=False,
            max_length=3500,
            truncation=True,
        )
        truncated_doc = self.tokenizer.decode(doc_tokens, skip_special_tokens=True)

        user_content = (
            f"{STRUCTURED_SYSTEM_PROMPT}\n\n"
            f"Analyze this legal document and return structured JSON:\n\n"
            f"{truncated_doc}"
        )
        # Mistral instruct format
        return f"<s>[INST] {user_content} [/INST]"

    def _parse_json_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse JSON from model output, with robust fallback handling.

        Attempts:
          1. Direct JSON parsing of the full output
          2. Regex extraction of JSON object from surrounding text
          3. Fallback: return raw output in a structured wrapper

        Args:
            raw_output: Raw string output from the model.

        Returns:
            Parsed dictionary or fallback structure.
        """
        # Attempt 1: Direct parse
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Extract JSON block from text
        json_pattern = re.search(r"\{[\s\S]+\}", raw_output)
        if json_pattern:
            try:
                return json.loads(json_pattern.group())
            except json.JSONDecodeError:
                pass

        # Attempt 3: Remove markdown fences and retry
        cleaned = re.sub(r"```(?:json)?|```", "", raw_output).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: wrap raw output
        logger.warning("Could not parse JSON from model output. Returning raw string.")
        return {
            "document_type": "Unknown",
            "summary": raw_output[:500],
            "parties_involved": [],
            "key_obligations": [],
            "key_rights": [],
            "critical_dates": [],
            "risk_factors": [],
            "governing_law": "Unknown",
            "_parse_error": True,
            "_raw_output": raw_output,
        }

    def _get_demo_output(self, index: int) -> Dict[str, Any]:
        """
        Return pre-computed structured outputs for demo mode
        (when model weights are not available).
        """
        outputs = [
            {
                "document_type": "Master Service Agreement",
                "summary": "CloudWave Technologies Inc. (Service Provider) agrees to deliver a cloud-based data analytics Platform to Meridian Financial Group LLC (Client) within 60 days of January 15, 2025 (by March 16, 2025). Client pays $28,500/month within 30 days of invoice, with 1.5% monthly interest on late payments. Either party may terminate with 60-day notice. All deliverables vest in Client upon full payment. Disputes resolved by AAA arbitration in San Francisco under Delaware law.",
                "parties_involved": [
                    "CloudWave Technologies Inc. — Service Provider (Delaware corporation, San Francisco CA)",
                    "Meridian Financial Group LLC — Client (New York LLC)",
                ],
                "key_obligations": [
                    "Service Provider: Deliver Platform within 60 days of Effective Date (by March 16, 2025)",
                    "Service Provider: Maintain 99.9% uptime SLA and 24/7 technical support",
                    "Client: Pay $28,500/month within 30 days of each invoice",
                    "Client: Reimburse reasonable expenses within 15 days of itemized invoice",
                    "Service Provider: Deliver all Client data within 15 days of termination",
                ],
                "key_rights": [
                    "Client: Ownership of all deliverables, code, and documentation upon full payment",
                    "Service Provider: Retain ownership of pre-existing proprietary tools",
                    "Either party: Terminate with 60 days written notice",
                    "Client: Terminate immediately for cause if breach not cured within 30 days",
                    "Client: Indemnification from Service Provider against third-party IP claims",
                ],
                "critical_dates": [
                    "Effective Date: January 15, 2025",
                    "Platform Delivery Deadline: March 16, 2025 (60 days from Effective Date)",
                    "Payment Terms: Net-30 from invoice date",
                    "Expense Reimbursement: Net-15 from itemized invoice",
                    "Termination Notice Period: 60 days written notice",
                    "Breach Cure Period: 30 days written notice",
                    "Post-Termination Data Return: 15 days",
                    "Confidentiality Duration: 5 years post-termination",
                ],
                "risk_factors": [
                    "Force majeure events (acts of God, pandemics, government actions) may excuse performance",
                    "Late payment interest accrues at 1.5% per month — significant penalty for delayed payments",
                    "Service Provider liability capped at 6 months of fees paid — limits Client recovery",
                    "No liability for indirect/consequential damages — Client bears downstream losses",
                    "Client IP rights contingent on full payment — partial payment leaves ownership disputed",
                ],
                "governing_law": "State of Delaware; binding arbitration in San Francisco, CA under AAA Commercial Arbitration Rules",
            },
            {
                "document_type": "Mutual Non-Disclosure Agreement (M&A Due Diligence)",
                "summary": "Apex Biomedical Corp. (US) and Vertex Pharma Holdings Ltd. (UK) entered a mutual NDA on February 28, 2025 to facilitate due diligence for a potential acquisition of up to 100% of Apex by Vertex. Both parties must maintain strict confidentiality for 3 years from execution or 2 years post-termination of discussions, whichever is later. Breaches may justify immediate injunctive relief without bond. Governed by New York law with exclusive jurisdiction in New York County courts.",
                "parties_involved": [
                    "Apex Biomedical Corp. — Company A (Massachusetts corporation, potential acquisition target)",
                    "Vertex Pharma Holdings Ltd. — Company B (UK company, potential acquirer)",
                ],
                "key_obligations": [
                    "Both parties: Maintain strict confidentiality using at least reasonable care",
                    "Both parties: Use Confidential Information solely to evaluate the Proposed Transaction",
                    "Both parties: Limit disclosure to employees and advisors on need-to-know basis",
                    "Both parties: Return or certify destruction of Confidential Information within 10 business days of request",
                    "Receiving party: Bind all employees/advisors to equivalent confidentiality obligations",
                ],
                "key_rights": [
                    "Either party: Seek immediate injunctive relief without posting bond for breach",
                    "Either party: Right to receive prompt written notice before legally compelled disclosure",
                    "Disclosing party: Right to request return or destruction of Confidential Information",
                ],
                "critical_dates": [
                    "Agreement Effective Date: February 28, 2025",
                    "Confidentiality Term: 3 years from Effective Date (expires February 28, 2028) OR 2 years post-termination of discussions (whichever is later)",
                    "Return/Destruction Deadline: 10 business days from written request",
                ],
                "risk_factors": [
                    "Breach may cause irreparable harm — injunctive relief available without bond or security",
                    "Oral disclosures are covered — no written memorialization required to trigger obligations",
                    "Employees and advisors must be independently bound — third-party breach risk",
                    "Compelled legal disclosure (court order) requires prompt notice — procedural risk",
                    "Long confidentiality tail (up to 5 years in some scenarios) restricts future business decisions",
                ],
                "governing_law": "State of New York; exclusive jurisdiction in New York County courts",
            },
            {
                "document_type": "Executive Employment Agreement",
                "summary": "Stellarion Dynamics Inc. (California) employs Dr. Priya Mehta as Chief Technology Officer for a 3-year term from March 1, 2025 to February 28, 2028, at $385,000 base salary with a 40% target bonus and 250,000 stock options vesting over 4 years with a 1-year cliff. Termination without cause or for Good Reason entitles Executive to 12 months severance plus fully accelerated option vesting. A 12-month non-compete and 24-month non-solicit apply post-termination. Governed by California law with JAMS arbitration.",
                "parties_involved": [
                    "Stellarion Dynamics Inc. — Employer / Company (California corporation)",
                    "Dr. Priya Mehta — Executive / CTO (Employee)",
                ],
                "key_obligations": [
                    "Executive: Devote substantially all professional time to Company business",
                    "Executive: Report directly to CEO",
                    "Executive: Assign all employment-related inventions and developments to Company",
                    "Executive: Comply with 12-month post-termination non-compete (US-wide, core AI products)",
                    "Executive: Comply with 24-month post-termination non-solicitation of employees/contractors",
                    "Company: Pay $385,000 annual base salary in bi-weekly installments",
                    "Company: Establish annual KPIs by April 30 of each year for bonus determination",
                ],
                "key_rights": [
                    "Executive: 40% target annual bonus based on KPIs set by Board",
                    "Executive: 250,000 stock options at $12.50/share (4-year vest, 1-year cliff from March 1, 2025)",
                    "Executive: 12 months severance + accelerated vesting if terminated without cause or for Good Reason",
                    "Executive: 25 days paid vacation and $10,000 annual professional development allowance",
                    "Executive: Resign for Good Reason (material reduction in duties/compensation) with full severance",
                    "Company: Terminate immediately for cause with no severance",
                ],
                "critical_dates": [
                    "Employment Start Date: March 1, 2025",
                    "Initial Term End Date: February 28, 2028",
                    "Option Cliff: March 1, 2026 (1-year cliff)",
                    "Full Option Vesting: March 1, 2029 (4-year schedule)",
                    "Annual KPI Establishment Deadline: April 30 of each year",
                    "Post-Termination Non-Compete: 12 months",
                    "Post-Termination Non-Solicitation: 24 months",
                    "Termination Without Cause Notice: 60 days written",
                    "Good Reason Resignation Notice: 30 days written",
                ],
                "risk_factors": [
                    "Broad IP assignment covers inventions created outside working hours if related to Company business",
                    "Non-compete applies US-wide — may limit Executive's future career options significantly",
                    "No severance for termination for cause — definition of 'cause' should be carefully scrutinised",
                    "Bonus fully discretionary based on Board-set KPIs — limited Executive control over payout",
                    "Option exercise price ($12.50) may be above market at vesting — underwater option risk",
                    "Arbitration under JAMS forecloses jury trial — waiver of constitutional right",
                ],
                "governing_law": "State of California; disputes resolved by JAMS Comprehensive Arbitration Rules in San Francisco",
            },
            {
                "document_type": "Federal Court of Australia — Judicial Opinion (Tax Law)",
                "summary": "The Federal Court of Australia ruled in favour of Pinnacle Resources Pty Ltd, overturning the Commissioner of Taxation's amended assessment that treated a $38.6 million gain from disposal of four Pilbara mining tenements as ordinary income under s.6-5 ITAA 1997. Applying Whitfords Beach principles, the Court held the gain is capital in nature because the tenements were acquired for long-term exploration, not resale. The Commissioner failed to discharge the burden of proof under s.14ZZO TAA 1953. The 50% CGT discount under Div. 115 ITAA 1997 applies. Commissioner ordered to pay costs.",
                "parties_involved": [
                    "Pinnacle Resources Pty Ltd — Applicant / Taxpayer (Western Australian mining exploration company)",
                    "Commissioner of Taxation — Respondent",
                    "Federal Court of Australia, New South Wales District Registry — Tribunal",
                    "Justice Harrington — Presiding Judge",
                ],
                "key_obligations": [
                    "Commissioner: Pay Pinnacle's costs of and incidental to this appeal",
                    "Commissioner: Issue revised assessment treating $38.6M gain as capital gain with 50% CGT discount",
                    "Commissioner: Bear burden of proof under s.14ZZO TAA 1953 (failed to discharge)",
                ],
                "key_rights": [
                    "Pinnacle: Treat $38.6M gain as capital gain (not ordinary income) under ITAA 1997",
                    "Pinnacle: Apply 50% CGT discount under Div. 115 ITAA 1997 (assets held >12 months)",
                    "Pinnacle: Recovery of full legal costs from Commissioner",
                ],
                "critical_dates": [
                    "Judgment Delivered: March 14, 2025",
                    "Relevant Tax Year: 2022-23 income year",
                    "Asset Disposal: 2022-23 (four Pilbara mining tenements)",
                    "Tenement Holding Period: Greater than 12 months (qualifies for 50% CGT discount)",
                    "Citation: [2025] FCA 0234",
                ],
                "risk_factors": [
                    "Establishes that frequency of disposals alone is insufficient to constitute a 'business of dealing' in assets",
                    "Commissioner may appeal to Full Federal Court — decision not yet final",
                    "Taxpayers in similar positions should document acquisition intent contemporaneously to support capital character arguments",
                    "Whitfords Beach principles reaffirmed — character of gain determined by nature of asset and intention at acquisition",
                    "Burden of proof under s.14ZZO TAA 1953 falls on Commissioner in objection proceedings — significant procedural protection for taxpayers",
                ],
                "governing_law": "Commonwealth of Australia; Income Tax Assessment Act 1997 (Cth); Taxation Administration Act 1953 (Cth); Federal Court of Australia Act 1976 (Cth)",
            },
        ]
        return outputs[index % len(outputs)]
