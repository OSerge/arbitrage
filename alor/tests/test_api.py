import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from unittest.mock import patch, Mock
from datetime import datetime
from alor.api import AlorAPI

@pytest.fixture
def api():
    return AlorAPI(token='test_token')

def test_init():
    # Тест инициализации без токена
    api = AlorAPI()
    assert api.base_url == "https://api.alor.ru/md/v2"
    assert 'Authorization' not in api.headers
    assert api.from_time is None
    assert api.to_time is None

    # Тест инициализации с токеном
    api = AlorAPI(token='test_token')
    assert api.headers['Authorization'] == 'Bearer test_token'

    # Тест инициализации с временными метками
    from_time = int(datetime(2024, 1, 1).timestamp())
    to_time = int(datetime(2024, 12, 31).timestamp())
    api = AlorAPI(from_time=from_time, to_time=to_time)
    assert api.from_time == from_time
    assert api.to_time == to_time

@patch('requests.get')
def test_get_security_historical_data(mock_get, api):
    # Настройка мока
    mock_response = Mock()
    mock_response.json.return_value = {'history': [{'data': 'test'}]}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # Тест успешного запроса
    result = api.get_security_historical_data(
        symbol='BRJ5',
        from_time='2024-01-01',
        to_time='2024-01-02'
    )
    assert result == [{'data': 'test'}]
    mock_get.assert_called_once()

    # Тест с неверным символом
    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        api.get_security_historical_data(symbol='')

    # Тест с неверным форматом времени
    with pytest.raises(ValueError, match="Invalid from_time format"):
        api.get_security_historical_data(
            symbol='BRJ5',
            from_time='invalid_date',
            to_time='2024-01-02'
        )

    # Тест с некорректным временным диапазоном
    with pytest.raises(ValueError, match="from_time must be less than to_time"):
        api.get_security_historical_data(
            symbol='BRJ5',
            from_time='2024-01-02',
            to_time='2024-01-01'
        )

@patch('requests.get')
def test_get_securities(mock_get, api):
    # Настройка мока
    mock_response = Mock()
    mock_response.json.return_value = ['BRJ5', 'SBRF']
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # Тест успешного запроса
    result = api.get_securities()
    assert result == ['BRJ5', 'SBRF']
    assert mock_get.call_count == 1
    mock_get.assert_called_with(
        'https://api.alor.ru/md/v2/Securities',
        headers={'Accept': 'application/json', 'Authorization': 'Bearer test_token'},
        params={
            'sector': 'FORTS',
            'exchange': 'MOEX',
            'instrumentGroup': 'RFUD',
            'limit': 50,
            'offset': 0
        }
    )

    # Тест с кастомными параметрами
    result = api.get_securities(
        sector='STOCKS',
        exchange='MOEX',
        instrument_group='TQBR',
        limit=100,
        offset=50
    )
    assert result == ['BRJ5', 'SBRF']
    assert mock_get.call_count == 2
    mock_get.assert_called_with(
        'https://api.alor.ru/md/v2/Securities',
        headers={'Accept': 'application/json', 'Authorization': 'Bearer test_token'},
        params={
            'sector': 'STOCKS',
            'exchange': 'MOEX',
            'instrumentGroup': 'TQBR',
            'limit': 100,
            'offset': 50
        }
    )

@patch('requests.get')
def test_error_handling(mock_get, api):
    # Тест обработки ошибки HTTP
    mock_get.side_effect = Exception('HTTP Error')
    
    with pytest.raises(Exception):
        api.get_security_historical_data(
            symbol='BRJ5',
            from_time='2024-01-01',
            to_time='2024-01-02'
        )

    with pytest.raises(Exception):
        api.get_securities()
