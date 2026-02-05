from codeshield.cs import CodeShield
import logging
from typing import Tuple, List, Optional
import json

class VulAnalyser():
    def __init__(self, llm_client=None, use_llm_verification=True):
        """
        Initialize vulnerability analyzer
        
        Args:
            llm_client: LLM client instance (e.g., QwenLLM) for secondary verification
            use_llm_verification: Whether to enable LLM verification, default True
        """
        self.llm_client = llm_client
        self.use_llm_verification = use_llm_verification
        
        # System prompt for LLM verification
        self.verification_prompt_template = """You are a code security expert. Please analyze whether the following code contains real security vulnerabilities.

Code:
```
{code}
```

The static analysis tool detected the following issues:
{issues}

Please carefully analyze:
1. Are these issues real security vulnerabilities?
2. Are there any false positives?
3. Does the code context indicate that these issues won't lead to actual security risks?

Please respond in JSON format:
{{
    "is_vulnerable": true/false,
    "confirmed_cwe_ids": ["CWE-XX", "CWE-YY"],
    "false_positives": ["CWE-ZZ"],
    "reasoning": "Your analysis and reasoning",
    "confidence": "high/medium/low"
}}

Focus on:
- Actual exploitability of the detected issues
- Context and usage patterns that might mitigate the risk
- Whether proper input validation or sanitization is present
- If the vulnerable code path is reachable
"""

    async def scan_code(self, code: str):
        """Scan code and return results"""
        try:
            result = await CodeShield.scan_code(code)
            return result
        except Exception as e:
            logging.error(f"CodeShield scan failed: {e}")
            raise

    def verify_with_llm(self, code: str, issues: List) -> Tuple[bool, List[str]]:
        """
        Verify static analysis results using LLM to reduce false positives
        
        Args:
            code: Code to analyze
            issues: List of issues detected by CodeShield
            
        Returns:
            (is_vulnerable, confirmed_cwe_ids)
        """
        if not self.llm_client or not self.use_llm_verification:
            # If LLM verification is not enabled, return static analysis results
            cwe_ids = []
            for issue in issues:
                if hasattr(issue, 'cwe_id') and issue.cwe_id:
                    cwe_ids.append(issue.cwe_id)
            return (True, cwe_ids)
        
        try:
            # Build issues description
            issues_description = "\n".join([
                f"- {getattr(issue, 'description', str(issue))} (CWE-{getattr(issue, 'cwe_id', 'Unknown')})"
                for issue in issues
            ])
            
            # Build verification prompt
            prompt = self.verification_prompt_template.format(
                code=code,
                issues=issues_description
            )
            
            # Call LLM
            logging.info(f"Requesting LLM verification for {len(issues)} issues")
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,  # Low temperature for more deterministic results
                max_tokens=2048,
                n=1
            )

            print("LLM Response:", response)
            
            # Parse LLM response
            try:
                # If response is a list, take the first element
                if isinstance(response, list):
                    response = response[0] if response else "{}"
                
                # Extract JSON part
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    logging.warning("LLM response does not contain valid JSON, using fallback")
                    result = {}
                
                is_vulnerable = result.get('is_vulnerable', True)
                confirmed_cwes = result.get('confirmed_cwe_ids', [])
                false_positives = result.get('false_positives', [])
                reasoning = result.get('reasoning', '')
                confidence = result.get('confidence', 'unknown')
                
                logging.info(f"LLM verification result: vulnerable={is_vulnerable}, "
                           f"confirmed={len(confirmed_cwes)}, "
                           f"false_positives={len(false_positives)}, "
                           f"confidence={confidence}")
                logging.debug(f"LLM reasoning: {reasoning}")
                
                # Only return vulnerability if confidence is not low
                if confidence == 'low' and is_vulnerable:
                    logging.warning("Low confidence vulnerability detection, treating as false positive")
                    return (False, [])
                
                return (is_vulnerable, confirmed_cwes)
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response as JSON: {e}")
                logging.debug(f"Raw response: {response}")
                # On JSON parse failure, conservatively return original static analysis results
                cwe_ids = [getattr(issue, 'cwe_id', '') for issue in issues 
                          if hasattr(issue, 'cwe_id') and issue.cwe_id]
                return (True, cwe_ids)
                
        except Exception as e:
            logging.error(f"LLM verification failed: {e}")
            # On LLM call failure, return static analysis results
            cwe_ids = [getattr(issue, 'cwe_id', '') for issue in issues 
                      if hasattr(issue, 'cwe_id') and issue.cwe_id]
            return (True, cwe_ids)

    async def predict(self, sequence):
        """
        Predict if code contains vulnerabilities, returns (label, cwe_ids)
        
        Process:
        1. Use CodeShield for static analysis
        2. If issues detected, use LLM for secondary verification
        3. Return comprehensive judgment result
        """
        try:
            # Step 1: Static analysis
            scan_result = await self.scan_code(sequence)
            
            logging.debug(f"Scan result: is_insecure={scan_result.is_insecure}, "
                         f"issues={len(scan_result.issues_found) if scan_result.issues_found else 0}")
            
            if not scan_result.is_insecure:
                return (0, [])
            
            # Step 2: LLM verification (only when static analysis finds issues)
            if scan_result.issues_found:
                for issue in scan_result.issues_found:
                    logging.debug(f"  Issue: {issue}")
                
                is_vulnerable, confirmed_cwes = self.verify_with_llm(
                    sequence, 
                    scan_result.issues_found
                )
                
                if is_vulnerable and confirmed_cwes:
                    logging.info(f"Confirmed vulnerabilities: {confirmed_cwes}")
                    return (1, list(set(confirmed_cwes)))
                else:
                    logging.info("LLM verification determined no real vulnerability (false positive)")
                    return (0, [])
            else:
                return (0, [])
                
        except Exception as e:
            logging.error(f"Exception while scanning for code security: {e}")
            # Default to safe on error
            return (0, [])