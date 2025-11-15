import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock, Mock
import sys
import pickle
from datetime import datetime

# Добавляем путь к исходному коду в Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.modeling.modeling import (
    check_models_by_nepv,
    create_full_pipeline,
    run_experiments,
    get_best_model_strategy,
    save_sorted_modeling_report,
    save_best_pipeline
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class TestCheckModelsByNepv:
    """Тесты для функции check_models_by_nepv"""
    
    @pytest.fixture
    def sample_regression_config(self):
        """Конфигурация для регрессии"""
        return {
            "data": {
                "logs": "logs"
            },
            "task_type": "regression"
        }
    
    @pytest.fixture
    def sample_classification_config(self):
        """Конфигурация для классификации"""
        return {
            "data": {
                "logs": "logs"
            },
            "task_type": "classif"
        }
    
    @pytest.fixture
    def sample_X_high_nepv(self):
        """Признаки с высоким NEPV"""
        return pd.DataFrame({
            'feature1': range(1000),
            'feature2': range(1000),
            'feature3': range(1000),
            'feature4': range(1000),
            'feature5': range(1000)
        })
    
    @pytest.fixture
    def sample_X_low_nepv(self):
        """Признаки с низким NEPV"""
        return pd.DataFrame({
            'feature1': range(10),
            'feature2': range(10),
            'feature3': range(10),
            'feature4': range(10),
            'feature5': range(10)
        })
    
    @pytest.fixture
    def sample_y_balanced(self):
        """Сбалансированные метки для классификации"""
        return pd.Series([0, 1] * 50)
    
    @pytest.fixture
    def sample_y_imbalanced(self):
        """Несбалансированные метки для классификации"""
        return pd.Series([0] * 10 + [1] * 2)  # 10:2 соотношение
    
    def test_nepv_regression_high_samples(self, sample_X_high_nepv, sample_regression_config):
        """Тест регрессии с большим количеством наблюдений"""
        models = check_models_by_nepv(
            X=sample_X_high_nepv,
            config=sample_regression_config,
            verbose=False
        )
        
        # Проверяем, что все модели включены при высоком NEPV
        expected_models = ["LinearRegression", "Ridge", "Lasso", "RandomForestRegressor", 
                          "GradientBoostingRegressor", "XGBRegressor", "LGBMRegressor", 
                          "DecisionTreeRegressor"]
        
        for model in expected_models:
            assert model in models, f"Модель {model} должна быть включена при высоком NEPV"
    
    def test_nepv_regression_low_samples(self, sample_X_low_nepv, sample_regression_config):
        """Тест регрессии с малым количеством наблюдений"""
        models = check_models_by_nepv(
            X=sample_X_low_nepv,
            config=sample_regression_config,
            verbose=False
        )
        
        # При низком NEPV должны быть только простые модели и DecisionTree
        expected_models = ["DecisionTreeRegressor"]
        
        for model in expected_models:
            assert model in models, f"Модель {model} должна быть включена даже при низком NEPV"
    
    def test_nepv_classification_balanced(self, sample_X_high_nepv, sample_y_balanced, sample_classification_config):
        """Тест классификации со сбалансированными данными"""
        models = check_models_by_nepv(
            X=sample_X_high_nepv,
            y=sample_y_balanced,
            config=sample_classification_config,
            verbose=False
        )
        
        # Проверяем наличие DecisionTreeClassifier
        assert "DecisionTreeClassifier" in models
    
    def test_nepv_classification_imbalanced(self, sample_X_low_nepv, sample_y_imbalanced, sample_classification_config):
        """Тест классификации с несбалансированными данными"""
        models = check_models_by_nepv(
            X=sample_X_low_nepv,
            y=sample_y_imbalanced,
            config=sample_classification_config,
            verbose=False
        )
        
        # При низком NEPV должны быть только простые модели
        assert "DecisionTreeClassifier" in models
    
    def test_nepv_unknown_task_type(self, sample_X_high_nepv, sample_regression_config):
        """Тест с неизвестным типом задачи"""
        invalid_config = sample_regression_config.copy()
        invalid_config["task_type"] = "unknown"
        
        with pytest.raises(ValueError, match="Неизвестный тип задачи"):
            check_models_by_nepv(
                X=sample_X_high_nepv,
                config=invalid_config,
                verbose=False
            )
    
    @patch('builtins.print')
    def test_nepv_verbose_true(self, mock_print, sample_X_high_nepv, sample_regression_config):
        """Тест вывода в консоль при verbose=True"""
        check_models_by_nepv(
            X=sample_X_high_nepv,
            config=sample_regression_config,
            verbose=True
        )
        
        # Проверяем, что print был вызван
        assert mock_print.called
    
    @patch('builtins.open')
    @patch('os.makedirs')
    def test_nepv_save_log(self, mock_makedirs, mock_open, sample_X_high_nepv, sample_regression_config):
        """Тест сохранения лога"""
        check_models_by_nepv(
            X=sample_X_high_nepv,
            config=sample_regression_config,
            verbose=False,
            save_log=True
        )
        
        # Проверяем, что директория создана и файл открыт для записи
        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()


class TestCreateFullPipeline:
    """Тесты для функции create_full_pipeline"""
    
    @pytest.fixture
    def sample_feature_config(self):
        """Пример конфигурации признаков"""
        return {
            'numerical_features': ['feature1', 'feature2'],
            'categorical_features': ['category'],
            'target_col': 'target'
        }
    
    @pytest.fixture
    def sample_model(self):
        """Пример модели"""
        return LinearRegression()
    
    @pytest.fixture
    def sample_outlier_strategy(self):
        """Пример стратегии обработки выбросов"""
        return ['iqr', 'zscore']
    
    def test_create_pipeline_structure(self, sample_outlier_strategy, sample_model, sample_feature_config):
        """Тест структуры созданного пайплайна"""
        pipeline = create_full_pipeline(
            outlier_strategy=sample_outlier_strategy,
            model_type='regression',
            model=sample_model,
            feature_config=sample_feature_config,
            y_name='target'
        )
        
        # Проверяем, что это Pipeline
        assert isinstance(pipeline, Pipeline)
        
        # Проверяем наличие основных шагов
        assert 'preprocess' in pipeline.named_steps
        assert 'standard_scaler' in pipeline.named_steps
        assert 'model' in pipeline.named_steps
    
    def test_create_pipeline_model_type(self, sample_outlier_strategy, sample_model, sample_feature_config):
        """Тест создания пайплайна с разными типами моделей"""
        for model_type in ['regression', 'classification']:
            pipeline = create_full_pipeline(
                outlier_strategy=sample_outlier_strategy,
                model_type=model_type,
                model=sample_model,
                feature_config=sample_feature_config,
                y_name='target'
            )
            
            assert isinstance(pipeline, Pipeline)
    
    def test_create_pipeline_different_models(self, sample_outlier_strategy, sample_feature_config):
        """Тест создания пайплайна с разными моделями"""
        models = [
            LinearRegression(),
            Ridge(),
            DecisionTreeRegressor(),
            RandomForestRegressor(n_estimators=10)
        ]
        
        for model in models:
            pipeline = create_full_pipeline(
                outlier_strategy=sample_outlier_strategy,
                model_type='regression',
                model=model,
                feature_config=sample_feature_config,
                y_name='target'
            )
            
            # Проверяем, что модель правильно встроена в пайплайн
            assert pipeline.named_steps['model'] == model


class TestRunExperiments:
    """Тесты для функции run_experiments"""
    
    @pytest.fixture
    def sample_data(self):
        """Пример данных для экспериментов"""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'W/C': np.random.uniform(0.3, 0.7, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        y = pd.Series(np.random.normal(0, 1, 100), name='target')
        return X, y
    
    @pytest.fixture
    def sample_strategies(self):
        """Пример стратегий обработки выбросов"""
        return [['iqr'], ['zscore']]
    
    @pytest.fixture
    def sample_model_types(self):
        """Пример типов моделей"""
        return ['regression']
    
    @pytest.fixture
    def sample_feature_config(self):
        """Пример конфигурации признаков"""
        return {
            'numerical_features': ['feature1', 'feature2', 'W/C'],
            'categorical_features': ['category'],
            'target_col': 'target'
        }
    
    @pytest.fixture
    def sample_models(self):
        """Пример моделей для тестирования"""
        return {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(max_depth=3, random_state=42)
        }
    
    @pytest.fixture
    def sample_config(self):
        """Пример конфигурации"""
        return {
            'outlier_detection': {
                'iqr_multiplier': 1.5,
                'zscore_threshold': 3
            },
            'task_type': 'regression'
        }
    
    @patch('src.modeling.modeling.OutlierHandler')
    @patch('src.modeling.modeling.cross_val_score')
    def test_run_experiments_basic(self, mock_cross_val, mock_outlier_handler, 
                                  sample_data, sample_strategies, sample_model_types,
                                  sample_feature_config, sample_models, sample_config):
        """Базовый тест запуска экспериментов"""
        # Мокаем OutlierHandler
        mock_handler_instance = Mock()
        mock_handler_instance.fit_transform.return_value = sample_data
        mock_outlier_handler.return_value = mock_handler_instance
        
        # Мокаем cross_val_score
        mock_cross_val.return_value = np.array([-0.5, -0.6, -0.55, -0.52, -0.58])
        
        X, y = sample_data
        
        results = run_experiments(
            X=X,
            y=y,
            all_strategies=sample_strategies,
            model_types=sample_model_types,
            feature_config=sample_feature_config,
            models=sample_models,
            config=sample_config,
            y_name='target'
        )
        
        # Проверяем, что результат - DataFrame
        assert isinstance(results, pd.DataFrame)
        
        # Проверяем наличие ожидаемых колонок
        expected_columns = [
            'outlier_strategy', 'model_name', 'model_type', 'dataset_size',
            'mean_rmse', 'std_rmse', 'training_time_sec', 'memory_used_mb'
        ]
        
        for col in expected_columns:
            assert col in results.columns
        
        # Проверяем, что были вызваны моки
        assert mock_outlier_handler.called
        assert mock_cross_val.called
    
    @patch('src.modeling.modeling.OutlierHandler')
    @patch('src.modeling.modeling.cross_val_score')
    def test_run_experiments_error_handling(self, mock_cross_val, mock_outlier_handler,
                                           sample_data, sample_strategies, sample_model_types,
                                           sample_feature_config, sample_models, sample_config):
        """Тест обработки ошибок в экспериментах"""
        # Мокаем OutlierHandler
        mock_handler_instance = Mock()
        mock_handler_instance.fit_transform.return_value = sample_data
        mock_outlier_handler.return_value = mock_handler_instance
        
        # Мокаем cross_val_score чтобы вызвать ошибку для одной модели
        def side_effect_cross_val(*args, **kwargs):
            model_name = args[0].named_steps['model'].__class__.__name__
            if model_name == 'LinearRegression':
                return np.array([-0.5, -0.6, -0.55])
            else:
                raise ValueError("Test error")
        
        mock_cross_val.side_effect = side_effect_cross_val
        
        X, y = sample_data
        
        results = run_experiments(
            X=X,
            y=y,
            all_strategies=sample_strategies,
            model_types=sample_model_types,
            feature_config=sample_feature_config,
            models=sample_models,
            config=sample_config,
            y_name='target'
        )
        
        # Проверяем, что эксперименты продолжались несмотря на ошибку
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0  # Должны быть успешные эксперименты


class TestGetBestModelStrategy:
    """Тесты для функции get_best_model_strategy"""
    
    @pytest.fixture
    def sample_modeling_result(self):
        """Пример результатов моделирования"""
        return pd.DataFrame({
            'model_name': ['ModelA', 'ModelB', 'ModelC'],
            'model_type': ['regression', 'regression', 'regression'],
            'outlier_strategy': ["['iqr']", "['zscore']", "['iqr']"],
            'mean_rmse': [1.5, 1.2, 1.8],
            'std_rmse': [0.1, 0.2, 0.15],
            'dataset_size': [100, 95, 98],
            'training_time_sec': [10.5, 15.2, 8.7],
            'memory_used_mb': [50.1, 60.3, 45.8]
        })
    
    def test_get_best_model_strategy(self, sample_modeling_result):
        """Тест выбора лучшей модели"""
        best_model_info = get_best_model_strategy(sample_modeling_result)
        
        # Проверяем структуру результата
        expected_keys = ['model', 'model_type', 'outlier_strategy', 'rmse']
        for key in expected_keys:
            assert key in best_model_info
        
        # Проверяем, что выбрана модель с минимальным RMSE
        assert best_model_info['model'] == 'ModelB'
        assert best_model_info['rmse'] == 1.2
    
    def test_get_best_model_strategy_empty_input(self):
        """Тест с пустым DataFrame"""
        empty_df = pd.DataFrame()
        
        # Вместо ValueError ожидаем KeyError, так как функция пытается получить колонку из пустого DF
        with pytest.raises((KeyError, ValueError)):
            get_best_model_strategy(empty_df)
    
    def test_get_best_model_strategy_missing_columns(self):
        """Тест с DataFrame без нужных колонок"""
        invalid_df = pd.DataFrame({
            'wrong_column': [1, 2, 3]
        })
        
        with pytest.raises(KeyError):
            get_best_model_strategy(invalid_df)


class TestSaveSortedModelingReport:
    """Тесты для функции save_sorted_modeling_report"""
    
    @pytest.fixture
    def sample_modeling_result(self):
        """Пример результатов для сохранения"""
        return pd.DataFrame({
            'model_name': ['ModelA', 'ModelB', 'ModelC'],
            'mean_rmse': [1.5, 1.2, 1.8],
            'std_rmse': [0.1, 0.2, 0.15]
        })
    
    @pytest.fixture
    def sample_config(self):
        """Пример конфигурации"""
        return {
            'models_saving': {
                'modeling_report_dir': 'reports/modeling'
            }
        }
    
    @patch('pandas.DataFrame.to_csv')
    @patch('builtins.print')
    def test_save_sorted_modeling_report(self, mock_print, mock_to_csv, 
                                        sample_modeling_result, sample_config):
        """Тест сохранения отсортированного отчета"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Мокаем Path чтобы использовать временную директорию
            with patch('src.modeling.modeling.Path') as mock_path:
                mock_path.return_value.resolve.return_value.parent.parent.parent = Path(temp_dir)
                
                save_sorted_modeling_report(
                    config=sample_config,
                    modeling_result=sample_modeling_result
                )
        
        # Проверяем, что to_csv был вызван
        mock_to_csv.assert_called_once()
        
        # Проверяем вывод сообщения
        mock_print.assert_called_once()
        assert "Отчет сохранен" in mock_print.call_args[0][0]


class TestSaveBestPipeline:
    """Тесты для функции save_best_pipeline"""
    
    @pytest.fixture
    def sample_modeling_result(self):
        """Пример результатов моделирования"""
        return pd.DataFrame({
            'model_name': ['LinearRegression', 'RandomForest'],
            'model_type': ['regression', 'regression'],
            'outlier_strategy': ["['iqr']", "['zscore']"],
            'mean_rmse': [1.2, 1.5],
            'std_rmse': [0.1, 0.2],
            'dataset_size': [100, 95],
            'training_time_sec': [10.5, 15.2],
            'memory_used_mb': [50.1, 60.3]
        })
    
    @pytest.fixture
    def sample_models(self):
        """Пример моделей"""
        return {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
    
    @pytest.fixture
    def sample_feature_config(self):
        """Пример конфигурации признаков"""
        return {
            'numerical_features': ['feature1', 'feature2'],
            'categorical_features': ['category'],
            'target_col': 'target'
        }
    
    @pytest.fixture
    def sample_config(self):
        """Пример конфигурации"""
        return {
            'models_saving': {
                'best_pipeline_dir': 'models/best'
            },
            'task_type': 'regression'
        }
    
    @patch('src.modeling.modeling.pickle.dump')
    @patch('builtins.open')
    @patch('builtins.print')
    def test_save_best_pipeline(self, mock_print, mock_open, mock_pickle_dump,
                               sample_modeling_result, sample_models, 
                               sample_feature_config, sample_config):
        """Тест сохранения лучшего пайплайна"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Мокаем Path чтобы использовать временную директорию
            with patch('src.modeling.modeling.Path') as mock_path:
                mock_path.return_value.resolve.return_value.parent.parent.parent = Path(temp_dir)
                
                save_best_pipeline(
                    best_model='LinearRegression',
                    best_model_type='regression',
                    best_outlier_strategy="['iqr']",
                    models=sample_models,
                    modeling_result=sample_modeling_result,
                    feature_config=sample_feature_config,
                    config=sample_config,
                    y_name='target'
                )
        
        # Проверяем, что pickle.dump был вызван дважды (пайплайн и метаданные)
        assert mock_pickle_dump.call_count == 2
        
        # Проверяем вывод сообщений
        assert mock_print.call_count >= 2
        print_messages = [call[0][0] for call in mock_print.call_args_list]
        assert any("Пайплайн сохранен" in msg for msg in print_messages)
        assert any("Лучшая модель" in msg for msg in print_messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])