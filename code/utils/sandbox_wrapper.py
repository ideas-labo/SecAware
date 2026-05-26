import logging
from typing import Tuple, Optional
from sandbox_fusion import RunCodeRequest, RunCodeResponse, RunStatus, run_code

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 15


def skipline(line: str) -> bool:
    skip_keywords = ['Traceback', 'File', 'line', '^\s*$']
    return any(keyword in line for keyword in skip_keywords)


def render_error(response: Optional[RunCodeResponse]) -> str:
    if response is None:
        return f"{_ERROR_MSG_PREFIX}Sandbox returned None (possible internal error)"
    
    error_parts = [_ERROR_MSG_PREFIX]
    
    if hasattr(response, 'compile_result') and response.compile_result:
        compile_result = response.compile_result
        error_parts.append("\n===== COMPILATION ERROR =====")
        
        if hasattr(compile_result, 'stderr') and compile_result.stderr:
            error_parts.append("STDERR:")
            error_parts.append(compile_result.stderr[:1500])
        
        if hasattr(compile_result, 'stdout') and compile_result.stdout:
            error_parts.append("STDOUT:")
            error_parts.append(compile_result.stdout[:1000])
        
        if hasattr(compile_result, 'exit_code'):
            error_parts.append(f"Exit Code: {compile_result.exit_code}")
    
    if hasattr(response, 'run_result') and response.run_result:
        run_result = response.run_result
        error_parts.append("\n===== RUNTIME ERROR =====")
        
        stdout = getattr(run_result, 'stdout', '')
        stderr = getattr(run_result, 'stderr', '')
        
        if stdout:
            error_parts.append("STDOUT:")
            error_parts.append(stdout[:1500])
        
        if stderr:
            error_parts.append("STDERR:")
            error_parts.append(stderr[:1500])
        
        if hasattr(run_result, 'exit_code'):
            error_parts.append(f"Exit Code: {run_result.exit_code}")
    
    else:
        status = getattr(response, 'status', 'unknown')
        message = getattr(response, 'message', '')
        error_parts.append(f"\nStatus: {status}")
        if message:
            error_parts.append(f"Message: {message}")
    
    return "\n".join(error_parts)


def code_exec_sandbox_fusion(
    code: str,
    stdin: str = "",
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    language: str = "python",
    pytest: Optional[str] = None,
) -> Tuple[bool, str]:

    try:
        request = RunCodeRequest(
            language=language,
            code=code,
            stdin=stdin,
            timeout=timeout,
            pytest_code=pytest if pytest else None
        )
        
        response = run_code(request)

        if response.status != RunStatus.Success:
            error_msg = render_error(response)
            return False, error_msg
        
        if hasattr(response, 'run_result') and response.run_result:
            stdout = getattr(response.run_result, 'stdout', '')
            stderr = getattr(response.run_result, 'stderr', '')
            
            if stdout:
                return True, stdout
            elif stderr:
                return True, stderr
        
        return True, ""
    
    except Exception as e:
        return False, f"{_ERROR_MSG_PREFIX}Exception: {type(e).__name__}: {str(e)}"

# def render_error(response: Optional[RunCodeResponse]) -> str:
#     """渲染错误信息，处理 None 情况"""
#     if response is None:
#         return f"{_ERROR_MSG_PREFIX}Sandbox returned None (possible internal error)"
    
#     if not hasattr(response, 'run_result') or response.run_result is None:
#         return f"{_ERROR_MSG_PREFIX}No run_result in response (status: {getattr(response, 'status', 'unknown')})"
    
#     stdout = getattr(response.run_result, 'stdout', '')
#     stderr = getattr(response.run_result, 'stderr', '')
    
#     log = (
#         _ERROR_MSG_PREFIX
#         + "\n===== STDOUT =====\n"
#         + (stdout or "(empty)")
#         + "\n===== STDERR =====\n"
#         + (stderr or "(empty)")
#     )
#     return "\n".join(l for l in log.split("\n") if not skipline(l))


def preprocess_java_code(code: str) -> str:

    if 'public class' in code and 'public static void main' in code:
        return code
    
    return code


def code_exec_sandbox_fusion(
    code: str,
    stdin: str = None,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    pytest: str = None,
    language: str = "python"
) -> Tuple[bool, str]:

    try:
        if language == "java":
            code = preprocess_java_code(code)
        
        if pytest:
            pytest_without_import = "\n".join(
                line
                for line in pytest.split("\n")
                if not line.startswith("from solution import")
            )
            
            request = RunCodeRequest(
                code=code + "\n" + pytest_without_import,
                timeout=timeout,
                language="pytest",
            )
        else:
            request = RunCodeRequest(
                code=code,
                stdin=stdin,
                timeout=timeout,
                language=language
            )

        response = run_code(request)
        
        if response is None:
            error_msg = render_error(None)
            logging.error(f"Sandbox returned None for language={language}")
            return False, error_msg
        
        if response.status != RunStatus.Success:
            return False, render_error(response)
        
        if not hasattr(response, 'run_result') or response.run_result is None:
            error_msg = render_error(response)
            logging.error(f"No run_result in response for language={language}")
            return False, error_msg
        
        stdout = getattr(response.run_result, 'stdout', '')
        return True, stdout if stdout else ""
        
    except AttributeError as e:
        error_msg = f"{_ERROR_MSG_PREFIX}AttributeError: {str(e)}\nPossible sandbox API incompatibility"
        logging.error(error_msg)
        return False, error_msg
    
    except Exception as e:
        error_msg = f"{_ERROR_MSG_PREFIX}Unexpected error: {type(e).__name__}: {str(e)}"
        logging.error(error_msg)
        return False, error_msg