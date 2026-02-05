import json
import os
from typing import Optional, Dict, List
import pickle
import logging

class KnowledgeBase:
    _instance: Optional['KnowledgeBase'] = None
    _data: Optional[Dict] = None
    
    def __new__(cls, filename: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, filename: str = None):
        if self._data is None and filename is not None:
            self.filename = filename
            self._data = self._load_with_cache()
    
    def _load_with_cache(self):
        # 生成 pickle 缓存文件路径
        pickle_file = self.filename.rsplit('.', 1)[0] + '.pkl'
        json_mtime = os.path.getmtime(self.filename) if os.path.exists(self.filename) else 0
        pickle_mtime = os.path.getmtime(pickle_file) if os.path.exists(pickle_file) else 0
        
        if os.path.exists(pickle_file) and pickle_mtime >= json_mtime:
            try:
                logging.info(f"Loading knowledge base from cache: {pickle_file}")
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                logging.info(f"Successfully loaded {len(data)} CWE entries from cache")
                return data
            except Exception as e:
                logging.warning(f"Failed to load pickle cache: {e}, falling back to JSON")

        logging.info(f"Loading knowledge base from JSON: {self.filename}")
        data = self.load_data()
        
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Created pickle cache: {pickle_file}")
        except Exception as e:
            logging.warning(f"Failed to create pickle cache: {e}")
        
        return data
    
    def load_data(self):
        """🔥 加载新格式的 cwe_knowledge_base.json"""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Knowledge base file not found: {self.filename}")
        
        with open(self.filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            result = {}
            for item in data:
                cwe_id = item.get("cwe_id")
                if cwe_id:
                    result[cwe_id] = item
            logging.info(f"Loaded {len(result)} CWE entries from JSON")
            return result
        
        logging.info(f"Loaded {len(data)} CWE entries from JSON")
        return data
    
    @classmethod
    def from_pickle(cls, pickle_file: str):
        if not os.path.exists(pickle_file):
            raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
        
        instance = cls.__new__(cls)
        with open(pickle_file, 'rb') as f:
            instance._data = pickle.load(f)
        logging.info(f"Loaded {len(instance._data)} CWE entries from pickle")
        return instance
    
    def save_pickle(self, pickle_file: str):
        if self._data is None:
            raise ValueError("No data to save")
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved {len(self._data)} CWE entries to {pickle_file}")
    
    def get_cwe(self, cwe_id: str) -> Optional[Dict]:
        if self._data is None:
            return None
        return self._data.get(cwe_id)
    
    def get_entrys(self, cwe_id: str) -> Dict:

        cwe_entry = self.get_cwe(cwe_id)
        if not cwe_entry:
            return self._get_empty_entry(cwe_id)
        
        if 'failure_modes' in cwe_entry:
            return {
                "cwe_id": cwe_entry.get("cwe_id", ""),
                "class": cwe_entry.get("class", ""),
                "failure_modes": cwe_entry.get("failure_modes", []),
                "preconditions": cwe_entry.get("preconditions", []),
                "sensitive_operations": cwe_entry.get("sensitive_operations", []),
                "typical_mitigations": cwe_entry.get("typical_mitigations", []),
                "semantic_risk": cwe_entry.get("semantic_risk", "")
            }
        else:
            return cwe_entry
    
    def _get_empty_entry(self, cwe_id: str) -> Dict:
        return {
            "cwe_id": cwe_id,
            "cwe_title": "",
            "description": "",
            "extended_description": "",
            "usage_scenarios": "",
            "design_challenges": "",
            "expanded_use_cases": "",
            "design_constraints": "",
            "engineering_tradeoffs": "",
            "external_dependencies": "",
            "industry_best_practices": "",
            "demonstrative_examples": {}
        }
    
    def get_failure_modes(self, cwe_id: str) -> List[str]:
        entry = self.get_cwe(cwe_id)
        return entry.get("failure_modes", []) if entry else []
    
    def get_preconditions(self, cwe_id: str) -> List[str]:
        entry = self.get_cwe(cwe_id)
        return entry.get("preconditions", []) if entry else []
    
    def get_sensitive_operations(self, cwe_id: str) -> List[str]:
        entry = self.get_cwe(cwe_id)
        return entry.get("sensitive_operations", []) if entry else []
    
    def get_typical_mitigations(self, cwe_id: str) -> List[str]:
        entry = self.get_cwe(cwe_id)
        return entry.get("typical_mitigations", []) if entry else []
    
    def get_semantic_risk(self, cwe_id: str) -> str:
        entry = self.get_cwe(cwe_id)
        return entry.get("semantic_risk", "") if entry else ""
    
    def get_cwes_by_class(self, class_name: str) -> List[str]:
        if self._data is None:
            return []
        
        return [
            cwe_id for cwe_id, entry in self._data.items()
            if entry.get("class", "").lower() == class_name.lower()
        ]
    
    def get_cwes_by_risk(self, semantic_risk: str) -> List[str]:
        if self._data is None:
            return []
        
        return [
            cwe_id for cwe_id, entry in self._data.items()
            if entry.get("semantic_risk", "").lower() == semantic_risk.lower()
        ]
    
    def get_all_cwe_ids(self) -> List[str]:
        if self._data is None:
            return []
        return list(self._data.keys())
    
    def get_statistics(self) -> Dict[str, any]:
        if self._data is None:
            return {}
        
        classes = {}
        risks = {}
        
        for entry in self._data.values():
            class_name = entry.get("class", "Unknown")
            classes[class_name] = classes.get(class_name, 0) + 1
            
            risk = entry.get("semantic_risk", "Unknown")
            risks[risk] = risks.get(risk, 0) + 1
        
        return {
            "total_cwes": len(self._data),
            "classes": classes,
            "semantic_risks": risks
        }
    
    @classmethod
    def clear_cache(cls):
        cls._instance = None
        cls._data = None


def load_knowledge_base(kb_path: str = None) -> KnowledgeBase:
    if kb_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        kb_path = os.path.join(base_dir, "../datasets/cwe_knowledge_distilled.json")
    
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
    
    return KnowledgeBase(filename=kb_path)