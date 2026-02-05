import re
import json
import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sandbox_wrapper import code_exec_sandbox_fusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    'c': 'cpp',
    'c++': 'cpp',
    'cpp': 'cpp',
    'python': 'python',
    'py': 'python',
    'java': 'java',
    'javascript': 'nodejs',
    'js': 'nodejs',
    'php': 'php',
    'csharp': 'csharp',  
    'cs': 'csharp',      
    'c#': 'csharp',      
}

def extract_code(solution_str: str, language: str) -> str:
    lang_markers = {
        'c': ['c', 'C'],
        'cpp': ['cpp', 'c++', 'C++'],
        'c++': ['cpp', 'c++', 'C++'],
        'python': ['python', 'py'],
        'java': ['java'],
        'javascript': ['javascript', 'js'],
        'php': ['php', 'PHP'],
        'csharp': ['csharp', 'cs', 'c#', 'C#'],  
        'cs': ['csharp', 'cs', 'c#', 'C#'],      
        'c#': ['csharp', 'cs', 'c#', 'C#'],     
    }
    
    markers = lang_markers.get(language.lower(), [language])
    for marker in markers:
        escaped_marker = re.escape(marker)
        pattern = rf'```{escaped_marker}\s*\n(.*?)```'
        matches = re.findall(pattern, solution_str, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
    
    pattern = r'```[a-zA-Z+#]*\s*\n(.*?)```'
    matches = re.findall(pattern, solution_str, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    if language.lower() == 'php' and solution_str.strip().startswith('<?php'):
        return solution_str.strip()
    
    if language.lower() in ('csharp', 'cs', 'c#'):
        if any(kw in solution_str for kw in ['using System', 'namespace ', 'class ', 'static void Main']):
            return solution_str.strip()
    
    return solution_str.strip()

def normalize_output(text: str) -> str:
    lines = text.strip().split('\n')
    normalized = []
    for line in lines:
        line = ' '.join(line.split())
        if line:
            normalized.append(line)
    return '\n'.join(normalized)


class FunctionalityValidator:
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def _run_single_test(
        self, 
        code: str, 
        stdin: str, 
        expected: str, 
        language: str
    ) -> Tuple[bool, str, str]:

        try:
            sandbox_lang = LANGUAGE_MAP.get(language.lower(), 'python')
            

            if language.lower() == 'php':
                code = self._preprocess_php_code(code)

            if language.lower() in ('csharp', 'cs', 'c#'):
                code = self._preprocess_csharp_code(code)
            
            success, output = code_exec_sandbox_fusion(
                code,
                stdin=stdin,
                timeout=self.timeout,
                language=sandbox_lang
            )
            
            if not success:
                return False, output, f"Execution failed: {output[:500]}"
            
            actual_norm = normalize_output(output)
            expected_norm = normalize_output(expected)
            
            if actual_norm == expected_norm:
                return True, output, ""
            else:
                return False, output, (
                    f"Output mismatch:\n"
                    f"Expected: {expected_norm[:200]}\n"
                    f"Actual: {actual_norm[:200]}"
                )
                
        except Exception as e:
            return False, "", f"Exception: {str(e)}"
    
    def _preprocess_php_code(self, code: str) -> str:

        code = code.strip()
        if not code.startswith('<?php'):
            code = '<?php\n' + code
        return code
    
    def _preprocess_csharp_code(self, code: str) -> str:

        code = code.strip()
        
        has_using = 'using System' in code
        has_namespace = 'namespace ' in code
        has_class = 'class ' in code
        has_main = 'static void Main' in code or 'static int Main' in code
        
        if not has_using:
            code = 'using System;\n' + code
        
        if not has_class:
            code = f'''using System;
using System.Linq;

public class Program
{{
    public static void Main()
    {{
{self._indent_code(code, 8)}
    }}
}}
'''
        elif not has_main:
            code = code.replace('public class', 'public class Program')
            if 'class Program' in code:
                lines = code.split('\n')
                class_line_idx = -1
                for i, line in enumerate(lines):
                    if 'class Program' in line:
                        class_line_idx = i
                        break
                
                if class_line_idx >= 0:
                    for i in range(class_line_idx, len(lines)):
                        if '{' in lines[i]:
                            lines.insert(i + 1, '    public static void Main() { }')
                            break
                    code = '\n'.join(lines)
        
        return code
    
    def _indent_code(self, code: str, spaces: int) -> str:
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line 
                        for line in code.split('\n'))
    
    def validate_implementation(
        self,
        fixed_code: str,
        test_cases: Dict,
        original_prompt: str,
        language: str = "python"
    ) -> Tuple[bool, Dict]:

        self.logger.info("=" * 80)
        self.logger.info(f"Validating {language} code implementation")
        self.logger.info("=" * 80)
        
        code = extract_code(fixed_code, language)
        self.logger.info(f"Extracted code length: {len(code)} chars")
        
        fc_tests = test_cases.get('fc_tests', [])
        if not fc_tests:
            self.logger.error("No fc_tests found in test_cases")
            return False, {
                'syntax_valid': True,
                'fc_tests_passed': 0,
                'fc_tests_total': 0,
                'fc_tests_results': [],
                'execution_logs': [],
                'pass_rate': 0.0
            }
        
        test_items = []
        for block_idx, test_block in enumerate(fc_tests):
            inputs = test_block.get('inputs', [])
            outputs = test_block.get('outputs', [])
            
            if isinstance(inputs, str):
                inputs = [inputs]
            if isinstance(outputs, str):
                outputs = [outputs]
            
            if len(inputs) != len(outputs):
                self.logger.warning(
                    f"Block {block_idx}: input/output count mismatch ({len(inputs)} vs {len(outputs)})"
                )
                continue
            
            for i, (inp, out) in enumerate(zip(inputs, outputs)):
                test_items.append({
                    'test_index': len(test_items),
                    'block_index': block_idx,
                    'case_index': i,
                    'input': inp,
                    'expected': out
                })
        
        total_tests = len(test_items)
        self.logger.info(f"Running {total_tests} test cases...")
        
        all_results = []
        passed_count = 0
        execution_logs = []
        
        max_workers = min(8, total_tests)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_test, 
                    code, 
                    tc['input'], 
                    tc['expected'], 
                    language
                ): tc
                for tc in test_items
            }
            
            for future in as_completed(futures):
                tc = futures[future]
                try:
                    success, actual, error_msg = future.result()
                    
                    result = {
                        'test_index': tc['test_index'],
                        'block_index': tc['block_index'],
                        'case_index': tc['case_index'],
                        'input': tc['input'],
                        'expected': tc['expected'],
                        'actual': actual,
                        'passed': success,
                        'log': error_msg if not success else ""
                    }
                    
                    all_results.append(result)
                    if success:
                        passed_count += 1
                    
                    execution_logs.append(
                        f"Test {tc['test_index']+1}: {'PASS' if success else 'FAIL'}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Test {tc['test_index']} execution error: {e}")
                    all_results.append({
                        'test_index': tc['test_index'],
                        'block_index': tc['block_index'],
                        'case_index': tc['case_index'],
                        'input': tc['input'],
                        'expected': tc['expected'],
                        'actual': '',
                        'passed': False,
                        'log': str(e)
                    })
        
        all_results.sort(key=lambda x: x['test_index'])
        
        is_valid = (passed_count == total_tests and total_tests > 0)
        
        details = {
            'syntax_valid': True,
            'fc_tests_passed': passed_count,
            'fc_tests_total': total_tests,
            'fc_tests_results': all_results,
            'execution_logs': execution_logs,
            'pass_rate': passed_count / max(total_tests, 1)
        }
        
        self.logger.info(f"Result: {passed_count}/{total_tests} passed ({details['pass_rate']*100:.1f}%)")
        
        return is_valid, details


def validate_entry(generated_code: str, entry: Dict, timeout: int = 60) -> Dict:
    idx = entry.get('idx', -1)
    language = entry.get('language', 'c')
    cwe = entry.get('cwe_identifier', 'N/A')
    
    validator = FunctionalityValidator(timeout=timeout)
    
    is_valid, details = validator.validate_implementation(
        fixed_code=generated_code,
        test_cases={'fc_tests': entry.get('fc_tests', [])},
        original_prompt=entry.get('test_case_prompt', ''),
        language=language
    )
    result = {
        'idx': idx,
        'language': language,
        'cwe': cwe,
        'success': is_valid,
        'passed': details.get('fc_tests_passed', 0),
        'failed': details.get('fc_tests_total', 0) - details.get('fc_tests_passed', 0),
        'total': details.get('fc_tests_total', 0),
        'details': details.get('fc_tests_results', [])
    }
    
    return result

if __name__ == "__main__":
    print("=" * 80)
    print("Testing C code")
    print("=" * 80)
    
    test_code_c = """
#include <stdio.h>
int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    printf("%d\\n", a + b);
    return 0;
}
"""
    
    test_entry_c = {
        'idx': 0,
        'language': 'c',
        'cwe_identifier': 'TEST',
        'fc_tests': [
            {
                'inputs': ['1 2', '5 10', '100 200'],
                'outputs': ['3', '15', '300']
            }
        ]
    }
    
    result_c = validate_entry(test_code_c, test_entry_c)
    print(json.dumps(result_c, indent=2))
    
    print("\n" + "=" * 80)
    print("Testing PHP code")
    print("=" * 80)
    
    test_code_php = """
<?php
$input = trim(fgets(STDIN));
list($a, $b) = explode(' ', $input);
echo ($a + $b) . "\\n";
?>
"""
    
    test_entry_php = {
        'idx': 1,
        'language': 'php',
        'cwe_identifier': 'TEST-PHP',
        'fc_tests': [
            {
                'inputs': ['1 2', '5 10', '100 200'],
                'outputs': ['3', '15', '300']
            }
        ]
    }
    
    result_php = validate_entry(test_code_php, test_entry_php)
    print(json.dumps(result_php, indent=2))
    
    print("\n" + "=" * 80)
    print("Testing C# code")
    print("=" * 80)
    
    test_code_csharp = """
using System;

class Program
{
    static void Main()
    {
        var input = Console.ReadLine().Split(' ');
        int a = int.Parse(input[0]);
        int b = int.Parse(input[1]);
        Console.WriteLine(a + b);
    }
}
"""
    
    test_entry_csharp = {
        'idx': 2,
        'language': 'csharp',
        'cwe_identifier': 'TEST-CSHARP',
        'fc_tests': [
            {
                'inputs': ['1 2', '5 10', '100 200'],
                'outputs': ['3', '15', '300']
            }
        ]
    }
    
    result_csharp = validate_entry(test_code_csharp, test_entry_csharp)
    print(json.dumps(result_csharp, indent=2))
    
    print("\n" + "=" * 80)
    print("Testing with FunctionalityValidator class")
    print("=" * 80)
    
    validator = FunctionalityValidator(timeout=60)
    is_valid, details = validator.validate_implementation(
        fixed_code=test_code_csharp,
        test_cases={'fc_tests': test_entry_csharp['fc_tests']},
        original_prompt="Write a C# program that adds two numbers",
        language='csharp'
    )
    
    print(f"\nIs Valid: {is_valid}")
    print(f"Passed: {details['fc_tests_passed']}/{details['fc_tests_total']}")
    print(f"Pass Rate: {details['pass_rate']*100:.1f}%")
    print("=" * 80)