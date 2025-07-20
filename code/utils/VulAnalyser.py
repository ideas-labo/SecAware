from codeshield.cs import CodeShield


class VulAnalyser():
    async def scan_code(self, cls, code: str):
        result = await CodeShield.scan_code(code)
        return result

    async def predict(self, sequence):
        scan_result = await self.scan_code(self, sequence)
        
        if scan_result.is_insecure:
            # extrct cwe-ids（if exist）
            cwe_ids = set()
            if scan_result.issues_found:
                for issue in scan_result.issues_found:
                    if hasattr(issue, 'cwe_id') and issue.cwe_id:
                        cwe_ids.add(issue.cwe_id)
            return (1, list(cwe_ids))
        else:
            return (0, [])  