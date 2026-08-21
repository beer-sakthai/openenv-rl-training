import pytest
from unittest.mock import patch, mock_open
import json
import sys
import os

# Set environment variables to prevent import errors in eval_bench_peft.py
os.environ['SAK_MODELS'] = 'dummy_model'

# Mock heavy dependencies globally to avoid actual imports
from unittest.mock import MagicMock
sys.modules['datasets'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()

# Ensure we can import the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.eval_bench_peft import _peft_base_model

def test_peft_base_model_valid():
    """Test when the adapter config successfully provides a base model."""
    with patch('huggingface_hub.hf_hub_download') as mock_download:
        mock_download.return_value = 'dummy_path.json'

        mock_file_content = '{"base_model_name_or_path": "mock_base_model"}'
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            base_model = _peft_base_model('mock_repo_id')
            assert base_model == 'mock_base_model'

def test_peft_base_model_missing_base_model():
    """Test when adapter_config.json is missing the base_model_name_or_path key."""
    with patch('huggingface_hub.hf_hub_download') as mock_download:
        mock_download.return_value = 'dummy_path.json'

        mock_file_content = '{"some_other_key": "value"}'
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            with pytest.raises(ValueError, match="adapter_config.json has no base_model_name_or_path"):
                _peft_base_model('mock_repo_id')

def test_peft_base_model_missing_config():
    """Test when the adapter_config.json file cannot be downloaded."""
    with patch('huggingface_hub.hf_hub_download') as mock_download:
        mock_download.side_effect = Exception("File not found")

        base_model = _peft_base_model('mock_repo_id')
        assert base_model is None
