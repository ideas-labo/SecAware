import json

class KnowledgeBase:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_data()
    
    def load_data(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_cwe(self, cwe_id):
        # 如果 self.data 是列表，则遍历查找匹配项
        if isinstance(self.data, list):
            for entry in self.data:
                if entry.get("cwe_id") == cwe_id or entry.get("id") == cwe_id:
                    return entry
            return None
        return self.data.get(cwe_id)
    
    def get_entrys(self, cwe_id):
        cwe_entry = self.get_cwe(cwe_id)
        if not cwe_entry:
            return {}
        
        # 提取 unsafe code 示例
        unsafe_code = ""
        demos = cwe_entry.get("demonstrative_examples", {})
        if isinstance(demos, dict) and "unsafe_code" in demos:
            unsafe_code = demos["unsafe_code"].get("code", "")
        elif isinstance(demos, list):
            for item in demos:
                if "unsafe_code" in item:
                    unsafe_code = item["unsafe_code"].get("code", "")
                    break
        
        return {
            "cwe_id": cwe_entry.get("cwe_id") or cwe_entry.get("id", ""),
            "cwe_title": cwe_entry.get("cwe_title", ""),
            "description": cwe_entry.get("description", ""),
            "extended_description": cwe_entry.get("extended_description", ""),
            "usage_scenarios": cwe_entry.get("usage_scenarios", ""),
            "design_challenges": cwe_entry.get("design_challenges", ""),
            "engineering_tradeoffs": cwe_entry.get("engineering_tradeoffs", ""),
            "expanded_use_cases": cwe_entry.get("expanded_use_cases", ""),
            # 平铺 unsafe code 示例
            "demonstrative_examples_unsafe_code": unsafe_code
        }



# 示例流程
if __name__ == "__main__":
    pass
