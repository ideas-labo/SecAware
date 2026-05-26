from typing import Tuple, Dict, List
import logging
from openai import OpenAI
from utils.func_validator import FunctionalityValidator
from utils.vul_analyzer import VulAnalyser

class DashScopeLLMClient:
    """DashScope LLM客户端包装器"""
    
    def __init__(self, api_key: str, model: str = "qwen-max", base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048, n: int = 1) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code security expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
                
        except Exception as e:
            logging.error(f"DashScope API call failed: {e}")
            raise


class CodeEvaluator:
    
    def __init__(self, 
                 func_validator: FunctionalityValidator = None,
                 vul_analyser: VulAnalyser = None,
                 timeout: int = 15,
                 dashscope_api_key: str = "sk-",
                 use_llm_verification: bool = True):
        self.func_validator = func_validator or FunctionalityValidator(timeout=timeout)
        
        if vul_analyser is None:
            llm_client = DashScopeLLMClient(
                api_key=dashscope_api_key,
                model="qwen-max"
            )
            self.vul_analyser = VulAnalyser(
                llm_client=llm_client,
                use_llm_verification=use_llm_verification
            )
        else:
            self.vul_analyser = vul_analyser
    
    async def evaluate(
        self,
        code: str,
        test_cases: Dict,
        original_prompt: str,
        language: str = "python"
    ) -> Tuple[bool, bool, Dict]:
        results = {
            "language": language,
            "functional": {},
            "security": {},
            "overall": {}
        }
        
        try:
            is_functional, func_details = self.func_validator.validate_implementation(
                fixed_code=code,
                test_cases=test_cases,
                original_prompt=original_prompt,
                language=language
            )
            
            results["functional"] = {
                "passed": is_functional,
                "syntax_valid": func_details.get("syntax_valid", False),
                "tests_passed": func_details.get("fc_tests_passed", 0),
                "tests_total": func_details.get("fc_tests_total", 0),
                "pass_rate": func_details.get("fc_tests_passed", 0) / max(func_details.get("fc_tests_total", 1), 1),
                "test_results": func_details.get("fc_tests_results", []),
                "logs": func_details.get("execution_logs", [])
            }
        except Exception as e:
            logging.error(f"Functional evaluation failed: {e}")
            is_functional = False
            results["functional"] = {
                "passed": False,
                "error": str(e)
            }
        try:
            vul_label, found_cwes = await self.vul_analyser.predict(code)
            is_secure = (vul_label == 0)  # 0表示安全
            
            results["security"] = {
                "passed": is_secure,
                "is_vulnerable": not is_secure,
                "found_cwes": found_cwes,
                "cwe_count": len(found_cwes)
            }
        except Exception as e:
            logging.error(f"Security evaluation failed: {e}")
            is_secure = True  
            results["security"] = {
                "passed": True,
                "error": str(e),
                "note": "Security scan failed, defaulting to secure"
            }
        
        results["overall"] = {
            "functional_passed": is_functional,
            "security_passed": is_secure,
            "both_passed": is_functional and is_secure,
            "summary": self._generate_summary(is_functional, is_secure, results)
        }
        
        return is_functional, is_secure, results
    
    def _generate_summary(self, is_functional: bool, is_secure: bool, results: Dict) -> str:
        """生成评估摘要"""
        func_info = results.get("functional", {})
        sec_info = results.get("security", {})
        
        status_parts = []
        
        if is_functional:
            pass_rate = func_info.get("pass_rate", 0) * 100
            status_parts.append(f"✓ Functional ({pass_rate:.1f}% tests passed)")
        else:
            status_parts.append("✗ Functional (failed)")
        
        if is_secure:
            status_parts.append("✓ Secure")
        else:
            cwes = sec_info.get("found_cwes", [])
            status_parts.append(f"✗ Vulnerable ({len(cwes)} CWEs: {', '.join(cwes[:3])})")
        
        return " | ".join(status_parts)
    
    def format_results(self, results: Dict, verbose: bool = False) -> str:
        output = []
        output.append("=" * 80)
        output.append("CODE EVALUATION RESULTS")
        output.append("=" * 80)
        
        overall = results.get("overall", {})
        output.append(f"\nOverall: {overall.get('summary', 'N/A')}")
        output.append(f"Both Passed: {'✓ YES' if overall.get('both_passed') else '✗ NO'}")
        
        func = results.get("functional", {})
        output.append(f"\n{'Functional Correctness':-^80}")
        output.append(f"Status: {'✓ PASS' if func.get('passed') else '✗ FAIL'}")
        output.append(f"Syntax: {'✓ Valid' if func.get('syntax_valid') else '✗ Invalid'}")
        output.append(f"Tests: {func.get('tests_passed', 0)}/{func.get('tests_total', 0)} passed ({func.get('pass_rate', 0)*100:.1f}%)")
        
        if verbose and func.get('test_results'):
            output.append("\nTest Details:")
            for test in func.get('test_results', [])[:10]: 
                status = "✓" if test.get('passed') else "✗"
                output.append(f"  {status} Test {test.get('test_index')}: {test.get('log', '')[:80]}")
        
        sec = results.get("security", {})
        output.append(f"\n{'Security Analysis':-^80}")
        output.append(f"Status: {'✓ SECURE' if sec.get('passed') else '✗ VULNERABLE'}")
        
        if not sec.get('passed'):
            cwes = sec.get('found_cwes', [])
            output.append(f"Vulnerabilities Found: {len(cwes)}")
            if cwes:
                output.append(f"CWEs: {', '.join(cwes)}")
        
        output.append("=" * 80)
        return "\n".join(output)


async def evaluate_code(
    code: str,
    test_cases: Dict,
    original_prompt: str,
    language: str = "python",
    dashscope_api_key: str = "sk-",
    timeout: int = 15
) -> Tuple[bool, bool, Dict]:
    """
    Returns:
        (is_functional, is_secure, detailed_results)
    """
    evaluator = CodeEvaluator(
        timeout=timeout,
        dashscope_api_key=dashscope_api_key,
        use_llm_verification=True
    )
    
    return await evaluator.evaluate(code, test_cases, original_prompt, language)


if __name__ == "__main__":
    import asyncio
    import json
    import hashlib
    
    def compute_hash(password, salt):
        combined = password + salt
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    async def test_evaluator():
        # 测试代码1: 安全的代码
        safe_code = """
import hashlib

def get_password_hash(password, salt):
    if password is None or salt is None:
        return 'Error: Invalid input'
    if not isinstance(password, str) or not isinstance(salt, str):
        return 'Error: Invalid input'
    
    combined = password + salt
    hash_object = hashlib.sha256(combined.encode('utf-8'))
    return hash_object.hexdigest()
"""
        
        vulnerable_sql_code = """
import sqlite3

def get_user_data(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # SQL注入漏洞 - 直接拼接用户输入
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result
"""

        vulnerable_exec_code = """
def calculate_expression(expr):
    # 代码执行漏洞 - 直接执行用户输入
    result = eval(expr)
    return result
"""
        
        test_cases = [
            ("安全且功能正确的代码", safe_code),
            ("SQL注入漏洞代码", vulnerable_sql_code),
            ("代码执行漏洞 (eval)", vulnerable_exec_code),
        ]
        
        evaluator = CodeEvaluator()
        
        for idx, (description, code) in enumerate(test_cases, 1):
            print("\n" + "=" * 80)
            print(f"测试{idx}: {description}")
            print("=" * 80)
            
            simple_test = {
                "prompt": description,
                "language": "python",
                "fc_tests": [] 
            }
            
            try:
                is_func, is_sec, results = await evaluator.evaluate(
                    code=code,
                    test_cases={"fc_tests": simple_test.get("fc_tests", [])},
                    original_prompt=simple_test["prompt"],
                    language="python"
                )
                
                print(evaluator.format_results(results, verbose=False))
                print(f"\n结果: 功能={'✓' if is_func else '✗'} | 安全={'✓' if is_sec else '✗'}")
                
                if not is_sec:
                    sec_info = results.get("security", {})
                    print(f"发现的漏洞: {', '.join(sec_info.get('found_cwes', ['Unknown']))}")
                
            except Exception as e:
                print(f"评估失败: {e}")
                import traceback
                traceback.print_exc()
    
    asyncio.run(test_evaluator())