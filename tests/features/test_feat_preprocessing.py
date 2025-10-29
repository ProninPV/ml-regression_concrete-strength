import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.features.feat_preprocessing import (
    OutlierHandler, 
    ZeroBinaryEncoder, 
    FeatureHandleEngineering,
    FeatureUninformRemove,
    CollinearityReducer,
    FeatureTransformer
)


class TestOutlierHandler:
    """Тесты для класса OutlierHandler"""
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных"""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'Cement': np.random.normal(300, 50, n_samples),
            'Water': np.random.normal(180, 20, n_samples),
            'Fine Aggregate': np.random.normal(700, 100, n_samples),
            'Sp/C_pct': np.random.normal(0.015, 0.005, n_samples),
            'Age': np.random.normal(28, 10, n_samples),
            'Strength': np.random.normal(40, 10, n_samples)
        })
        
        # Добавляем несколько выбросов
        data.loc[10, 'Sp/C_pct'] = 0.1  # Аномальное значение
        data.loc[20, 'Fine Aggregate'] = 1000  # Аномальное значение
        data.loc[30, 'Cement'] = 50  # Низкое значение вне ГОСТ
        data.loc[40, 'Water'] = 300  # Высокое значение вне ГОСТ
        
        return data
    
    @pytest.fixture
    def outlier_handler(self):
        """Создание экземпляра OutlierHandler"""
        config = {
            'standard_value': {
                'cement': {'max': 600, 'min': 200},
                'water': {'max': 220, 'min': 120},
                'fine_aggregate': {'max': 800, 'min': 600},
                'sp_c_pct': {'max': 0.025, 'min': 0.005}
            },
            'abnormal_value': {
                'sp_c_pct': 0.07,
                'fine_aggregate': 970
            }
        }
        return OutlierHandler(
            strategies=['abnormal', 'gost_remove'],
            config=config,
            target_col='Strength'
        )
    
    def test_initialization(self, outlier_handler):
        """Тест инициализации класса"""
        assert outlier_handler.strategies == ['abnormal', 'gost_remove']
        assert 'cement' in outlier_handler.gost_ranges
        assert outlier_handler.target_col == 'Strength'
        assert not outlier_handler.fitted_
    
    def test_fit_method(self, outlier_handler, sample_data):
        """Тест метода fit"""
        X = sample_data.drop('Strength', axis=1)
        outlier_handler.fit(X)
        
        assert outlier_handler.fitted_
        assert outlier_handler.feature_names_in_ is not None
        assert 'original_shape' in outlier_handler.outlier_summary_
    
    def test_abnormal_strategy(self, sample_data):
        """Тест стратегии abnormal"""
        handler = OutlierHandler(strategies=['abnormal'])
        X = sample_data.drop('Strength', axis=1)
        y = sample_data['Strength']
        
        X_fit = X.copy()
        handler.fit(X_fit)
        X_transformed, y_transformed = handler.transform(X, y)
        
        # Проверяем, что аномальные строки удалены
        assert len(X_transformed) < len(X)
        assert 10 not in X_transformed.index  # Индекс с Sp/C_pct = 0.1
        assert 20 not in X_transformed.index  # Индекс с Fine Aggregate = 1000
    
    def test_gost_remove_strategy(self, sample_data):
        """Тест стратегии gost_remove"""
        handler = OutlierHandler(strategies=['gost_remove'])
        X = sample_data.drop('Strength', axis=1)
        
        handler.fit(X)
        X_transformed = handler.transform(X)
        
        # Проверяем, что значения вне ГОСТ удалены
        assert len(X_transformed) < len(X)
    
    def test_iqr_strategy(self, sample_data):
        """Тест стратегии iqr_remove"""
        handler = OutlierHandler(strategies=['iqr_remove'])
        X = sample_data.drop('Strength', axis=1)
        
        handler.fit(X)
        X_transformed = handler.transform(X)
        
        # Проверяем, что IQR границы рассчитаны
        assert 'Cement' in handler.iqr_bounds_
        assert len(X_transformed) <= len(X)
    
    def test_combine_strategy(self, sample_data):
        """Тест комбинированной стратегии"""
        handler = OutlierHandler(strategies=['combine'])
        X = sample_data.drop('Strength', axis=1)
        
        handler.fit(X)
        X_transformed = handler.transform(X)
        
        # Проверяем создание бинарных признаков
        assert 'High_SP' in X_transformed.columns or len(X_transformed) < len(X)
    
    def test_get_outlier_summary(self, outlier_handler, sample_data):
        """Тест получения сводки по выбросам"""
        X = sample_data.drop('Strength', axis=1)
        outlier_handler.fit(X)
        outlier_handler.transform(X)
        
        summary = outlier_handler.get_outlier_summary()
        
        assert 'original_shape' in summary
        assert 'removed_rows' in summary
        assert 'strategies_applied' in summary


class TestZeroBinaryEncoder:
    """Тесты для класса ZeroBinaryEncoder"""
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных с нулевыми значениями"""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'Blast Furnace Slag': np.random.choice([0, 50, 100], n_samples, p=[0.3, 0.4, 0.3]),
            'Fly Ash': np.random.choice([0, 30, 60], n_samples, p=[0.4, 0.3, 0.3]),
            'Sp/C_pct': np.random.choice([0, 0.01, 0.02], n_samples, p=[0.2, 0.4, 0.4]),
            'Cement': np.random.normal(300, 50, n_samples),
            'Strength': np.random.normal(40, 8, n_samples)
        })
        
        # Создаем различия в Strength для групп с нулевыми и ненулевыми значениями
        slag_non_zero_mask = data['Blast Furnace Slag'] > 0
        data.loc[slag_non_zero_mask, 'Strength'] += 5
        
        return data
    
    def test_initialization(self):
        """Тест инициализации класса"""
        encoder = ZeroBinaryEncoder(alpha=0.05, levene_alpha=0.05)
        assert encoder.alpha == 0.05
        assert encoder.levene_alpha == 0.05
        assert encoder.significant_features_ == []
    
    def test_fit_method(self, sample_data):
        """Тест метода fit"""
        encoder = ZeroBinaryEncoder(alpha=0.05)
        X = sample_data.drop('Strength', axis=1)
        y = sample_data['Strength']
        
        encoder.fit(X, y)
        
        assert len(encoder.test_results_) > 0
        assert 'Blast Furnace Slag' in encoder.test_results_
    
    def test_transform_method(self, sample_data):
        """Тест метода transform"""
        encoder = ZeroBinaryEncoder(alpha=0.05)
        X = sample_data.drop('Strength', axis=1)
        y = sample_data['Strength']
        
        encoder.fit(X, y)
        X_transformed = encoder.transform(X)
        
        # Проверяем, что добавлены бинарные признаки для значимых фич
        if encoder.significant_features_:
            for feature in encoder.significant_features_:
                expected_col = f"{feature}_binary"
                assert expected_col in X_transformed.columns
                assert X_transformed[expected_col].dtype in [int, bool]
    
    def test_fit_transform_method(self, sample_data):
        """Тест объединенного метода fit_transform"""
        encoder = ZeroBinaryEncoder(alpha=0.05)
        X = sample_data.drop('Strength', axis=1)
        y = sample_data['Strength']
        
        X_transformed = encoder.fit_transform(X, y)
        
        assert len(encoder.test_results_) > 0
        assert isinstance(X_transformed, pd.DataFrame)
    
    def test_get_summary(self, sample_data):
        """Тест получения сводки"""
        encoder = ZeroBinaryEncoder(alpha=0.05)
        X = sample_data.drop('Strength', axis=1)
        y = sample_data['Strength']
        
        encoder.fit(X, y)
        summary = encoder.get_summary()
        
        assert 'significant_features' in summary
        assert 'test_results' in summary
        assert 'alpha' in summary


class TestFeatureHandleEngineering:
    """Тесты для класса FeatureHandleEngineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных для инженерии признаков"""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'Cement': np.random.uniform(200, 400, n_samples),
            'Blast Furnace Slag': np.random.uniform(0, 150, n_samples),
            'Fly Ash': np.random.uniform(0, 100, n_samples),
            'Coarse Aggregate': np.random.uniform(800, 1200, n_samples),
            'Fine Aggregate': np.random.uniform(500, 800, n_samples),
            'Water': np.random.uniform(150, 200, n_samples),
            'Superplasticizer': np.random.uniform(0, 20, n_samples),
            'W/C': np.random.uniform(0.3, 0.6, n_samples)
        })
    
    def test_initialization(self):
        """Тест инициализации класса"""
        # Для линейных моделей
        engineer_linear = FeatureHandleEngineering(model_type='linear_models')
        assert engineer_linear.model_type == 'linear_models'
        
        # Для деревьев
        engineer_trees = FeatureHandleEngineering(model_type='trees_models')
        assert engineer_trees.model_type == 'trees_models'
    
    def test_fit_method(self, sample_data):
        """Тест метода fit"""
        engineer = FeatureHandleEngineering(model_type='trees_models')
        X = sample_data
        
        engineer.fit(X)
        
        assert engineer.low_wc_threshold_ is not None
        assert engineer.high_wc_threshold_ is not None
        assert engineer.feature_names_ is not None
    
    def test_transform_linear_models(self, sample_data):
        """Тест transform для линейных моделей"""
        engineer = FeatureHandleEngineering(model_type='linear_models')
        X = sample_data
        
        engineer.fit(X)
        X_transformed = engineer.transform(X)
        
        # Проверяем базовые признаки
        expected_features = ['Total_powder', 'Aggregate_ratio', 'Binder_water_ratio']
        for feature in expected_features:
            assert feature in X_transformed.columns
        
        # Проверяем, что признаки для деревьев не добавлены
        trees_features = ['Has_Slag', 'Has_FlyAsh', 'Low_WC_ratio']
        for feature in trees_features:
            assert feature not in X_transformed.columns
    
    def test_transform_trees_models(self, sample_data):
        """Тест transform для деревьев"""
        engineer = FeatureHandleEngineering(model_type='trees_models')
        X = sample_data
        
        engineer.fit(X)
        X_transformed = engineer.transform(X)
        
        # Проверяем базовые признаки
        base_features = ['Total_powder', 'Aggregate_ratio', 'Binder_water_ratio']
        for feature in base_features:
            assert feature in X_transformed.columns
        
        # Проверяем дополнительные признаки для деревьев
        trees_features = ['Has_Slag', 'Has_FlyAsh', 'Has_Superplasticizer']
        for feature in trees_features:
            assert feature in X_transformed.columns
        
        # Проверяем бинарные признаки
        binary_features = [col for col in X_transformed.columns if col.startswith('Has_')]
        assert len(binary_features) >= 3
    
    def test_get_feature_summary(self, sample_data):
        """Тест получения сводки по признакам"""
        engineer = FeatureHandleEngineering(model_type='trees_models')
        X = sample_data
        
        engineer.fit(X)
        summary = engineer.get_feature_summary()
        
        assert 'model_type' in summary
        assert 'total_features_created' in summary
        assert 'features_created' in summary
        assert 'wc_thresholds' in summary
        
        # Проверяем, что для trees моделей создано больше признаков
        assert summary['total_features_created'] > 3
    
    def test_get_feature_descriptions(self, sample_data):
        """Тест получения описаний признаков"""
        engineer = FeatureHandleEngineering(model_type='trees_models')
        
        descriptions = engineer.get_feature_descriptions()
        
        assert 'Total_powder' in descriptions
        assert 'Has_Slag' in descriptions
        assert isinstance(descriptions, dict)


class TestFeatureUninformRemove:
    """Тесты для класса FeatureUninformRemove"""
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных с неинформативными признаками"""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'informative_1': np.random.normal(0, 1, n_samples),
            'informative_2': np.random.normal(0, 1, n_samples),
            'uninformative_1': [1] * 98 + [0] * 2,  # 98% одного значения
            'uninformative_2': ['A'] * 96 + ['B'] * 4,  # 96% одного значения
            'borderline': [1] * 90 + [0] * 10  # 90% одного значения
        })
        
        return data
    
    def test_initialization(self):
        """Тест инициализации класса"""
        remover = FeatureUninformRemove(threshold=0.95, verbose=False)
        assert remover.threshold == 0.95
        assert remover.verbose == False
    
    def test_fit_method(self, sample_data):
        """Тест метода fit"""
        remover = FeatureUninformRemove(threshold=0.95, verbose=False)
        
        remover.fit(sample_data)
        
        assert len(remover.columns_to_drop_) > 0
        assert 'uninformative_1' in remover.columns_to_drop_
        assert 'uninformative_2' in remover.columns_to_drop_
        assert 'borderline' not in remover.columns_to_drop_  # 90% < 95%
    
    def test_transform_method(self, sample_data):
        """Тест метода transform"""
        remover = FeatureUninformRemove(threshold=0.95, verbose=False)
        
        remover.fit(sample_data)
        X_transformed = remover.transform(sample_data)
        
        # Проверяем, что неинформативные признаки удалены
        assert 'uninformative_1' not in X_transformed.columns
        assert 'uninformative_2' not in X_transformed.columns
        assert 'informative_1' in X_transformed.columns
        assert 'informative_2' in X_transformed.columns
        assert 'borderline' in X_transformed.columns
        
        # Проверяем размерность
        assert X_transformed.shape[1] == sample_data.shape[1] - 2
    
    def test_different_thresholds(self, sample_data):
        """Тест с разными порогами"""
        # Более строгий порог
        remover_strict = FeatureUninformRemove(threshold=0.85, verbose=False)
        remover_strict.fit(sample_data)
        
        # Более мягкий порог
        remover_lenient = FeatureUninformRemove(threshold=0.97, verbose=False)
        remover_lenient.fit(sample_data)
        
        # Строгий порог должен удалять больше признаков
        assert len(remover_strict.columns_to_drop_) >= len(remover_lenient.columns_to_drop_)
    
    def test_get_removal_summary(self, sample_data):
        """Тест получения сводки по удалению"""
        remover = FeatureUninformRemove(threshold=0.95, verbose=False)
        
        remover.fit(sample_data)
        summary = remover.get_removal_summary()
        
        assert 'threshold' in summary
        assert 'columns_removed' in summary
        assert 'removal_stats' in summary
        assert 'total_removed' in summary


class TestCollinearityReducer:
    """Тесты для класса CollinearityReducer"""
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных с мультиколлинеарностью"""
        np.random.seed(42)
        n_samples = 100
        
        base_feature = np.random.normal(0, 1, n_samples)
        
        data = pd.DataFrame({
            'feature_1': base_feature,
            'feature_2': base_feature + np.random.normal(0, 0.1, n_samples),  # Высокая корреляция
            'feature_3': base_feature * 2 + np.random.normal(0, 0.1, n_samples),  # Высокая корреляция
            'feature_4': np.random.normal(0, 1, n_samples),  # Независимый
            'feature_5': np.random.normal(0, 1, n_samples),  # Независимый
            'Water': np.random.normal(180, 20, n_samples),  # Защищенный признак
            'Cement': np.random.normal(300, 50, n_samples)  # Защищенный признак
        })
        
        return data
    
    def test_initialization(self):
        """Тест инициализации класса"""
        reducer = CollinearityReducer(
            vif_threshold=10,
            correlation_threshold=0.95,
            priority_strategy='domain_priority',
            protected_features=['Water', 'Cement'],
            verbose=False
        )
        
        assert reducer.vif_threshold == 10
        assert reducer.correlation_threshold == 0.95
        assert reducer.priority_strategy == 'domain_priority'
        assert 'Water' in reducer.protected_features
    
    def test_fit_method(self, sample_data):
        """Тест метода fit"""
        reducer = CollinearityReducer(verbose=False)
        
        reducer.fit(sample_data)
        
        assert len(reducer.final_features_) > 0
        assert len(reducer.removed_features_) >= 0
        assert hasattr(reducer, 'vif_report_')
    
    def test_transform_method(self, sample_data):
        """Тест метода transform"""
        reducer = CollinearityReducer(verbose=False)
        
        reducer.fit(sample_data)
        X_transformed = reducer.transform(sample_data)
        
        # Проверяем размерность
        assert X_transformed.shape[1] == len(reducer.final_features_)
        assert X_transformed.shape[0] == sample_data.shape[0]
        
        # Проверяем, что все финальные признаки присутствуют
        for feature in reducer.final_features_:
            assert feature in X_transformed.columns
    
    def test_protected_features(self, sample_data):
        """Тест защиты важных признаков"""
        reducer = CollinearityReducer(
            protected_features=['Water', 'Cement'],
            verbose=False
        )
        
        reducer.fit(sample_data)
        
        # Проверяем, что защищенные признаки не удалены
        assert 'Water' in reducer.final_features_
        assert 'Cement' in reducer.final_features_
    
    def test_get_removal_report(self, sample_data):
        """Тест получения отчета"""
        reducer = CollinearityReducer(verbose=False)
        
        reducer.fit(sample_data)
        report = reducer.get_removal_report()
        
        assert 'parameters' in report
        assert 'summary' in report
        assert 'removed_features' in report
        assert 'final_features' in report
        assert 'vif_analysis' in report


class TestFeatureTransformer:
    """Тесты для класса FeatureTransformer"""
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных для трансформации"""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'feature_1': np.random.lognormal(1, 1, n_samples),  # Логнормальное распределение
            'feature_2': np.random.chisquare(5, n_samples),  # Хи-квадрат распределение
            'feature_3': np.random.normal(0, 1, n_samples),  # Нормальное распределение
            'binary_feature': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # Бинарный признак
            'Age': np.random.lognormal(3, 1, n_samples),  # Для теста Log преобразования
            'Strength': np.random.normal(40, 10, n_samples)  # Целевая переменная
        })
        
        return data
    
    @pytest.fixture
    def transformer_config(self):
        """Конфигурация для трансформера"""
        return {
            'trend_settings': {
                'names': ['Linear', 'Log', 'Sqrt', '1/x', 'square_func']
            }
        }
    
    def test_initialization(self, transformer_config):
        """Тест инициализации класса"""
        transformer = FeatureTransformer(
            config=transformer_config,
            target_col='Strength',
            alpha=0.2
        )
        
        assert transformer.config == transformer_config
        assert transformer.target_col == 'Strength'
        assert transformer.alpha == 0.2
    
    def test_fit_method(self, sample_data, transformer_config):
        """Тест метода fit"""
        transformer = FeatureTransformer(
            config=transformer_config,
            target_col='Strength'
        )
        
        transformer.fit(sample_data)
        
        assert len(transformer.best_transformations_) > 0
        assert len(transformer.transformation_report_) > 0
    
    def test_transform_method(self, sample_data, transformer_config):
        """Тест метода transform"""
        transformer = FeatureTransformer(
            config=transformer_config,
            target_col='Strength'
        )
        
        transformer.fit(sample_data)
        X_transformed = transformer.transform(sample_data)
        
        # Проверяем, что данные преобразованы
        assert X_transformed.shape == sample_data.shape
        
        # Проверяем, что бинарные признаки не преобразованы
        binary_col_data = X_transformed['binary_feature']
        original_binary_data = sample_data['binary_feature']
        assert np.array_equal(binary_col_data, original_binary_data)
    
    def test_binary_feature_detection(self, sample_data, transformer_config):
        """Тест обнаружения бинарных признаков"""
        transformer = FeatureTransformer(
            config=transformer_config,
            target_col='Strength'
        )
        
        transformer.fit(sample_data)
        
        # Проверяем, что бинарный признак определен правильно
        assert 'binary_feature' in transformer.best_transformations_
        assert transformer.best_transformations_['binary_feature'] == 'Linear'
        
        # Проверяем отчет
        report = transformer.transformation_report_['binary_feature']
        assert report['is_binary'] == True
    
    def test_get_transformation_report(self, sample_data, transformer_config):
        """Тест получения отчета по трансформациям"""
        transformer = FeatureTransformer(
            config=transformer_config,
            target_col='Strength'
        )
        
        transformer.fit(sample_data)
        report_df = transformer.get_transformation_report()
        
        assert isinstance(report_df, pd.DataFrame)
        assert len(report_df) == len(transformer.best_transformations_)
        assert 'Признак' in report_df.columns
        assert 'Лучшее преобразование' in report_df.columns


class TestIntegration:
    """Интеграционные тесты для всего пайплайна"""
    
    @pytest.fixture
    def full_sample_data(self):
        """Полный набор тестовых данных"""
        np.random.seed(42)
        n_samples = 200
        
        data = pd.DataFrame({
            'Cement': np.random.normal(300, 50, n_samples),
            'Blast Furnace Slag': np.random.choice([0, 50, 100], n_samples, p=[0.3, 0.4, 0.3]),
            'Fly Ash': np.random.choice([0, 30, 60], n_samples, p=[0.4, 0.3, 0.3]),
            'Water': np.random.normal(180, 20, n_samples),
            'Superplasticizer': np.random.uniform(0, 15, n_samples),
            'Coarse Aggregate': np.random.normal(1000, 100, n_samples),
            'Fine Aggregate': np.random.normal(700, 80, n_samples),
            'Age': np.random.lognormal(3, 0.5, n_samples),
            'Sp/C_pct': np.random.normal(0.015, 0.005, n_samples),
            'Strength': np.random.normal(40, 8, n_samples)
        })
        
        # Добавляем W/C ratio
        data['W/C'] = data['Water'] / data['Cement']
        
        return data
    
    def test_full_pipeline(self, full_sample_data):
        """Тест полного пайплайна обработки"""
        # Шаг 1: Обработка выбросов
        outlier_config = {
            'standard_value': {
                'cement': {'max': 600, 'min': 200},
                'water': {'max': 220, 'min': 120},
                'fine_aggregate': {'max': 800, 'min': 600}
            },
            'abnormal_value': {
                'sp_c_pct': 0.07,
                'fine_aggregate': 970
            }
        }
        
        outlier_handler = OutlierHandler(
            strategies=['abnormal'],
            config=outlier_config,
            target_col='Strength'
        )
        
        X = full_sample_data.drop('Strength', axis=1)
        y = full_sample_data['Strength']
        
        X_clean, y_clean = outlier_handler.fit_transform(X, y)
        
        # Шаг 2: Инженерия признаков
        feature_engineer = FeatureHandleEngineering(model_type='trees_models')
        X_engineered = feature_engineer.fit_transform(X_clean)
        
        # Шаг 3: Добавление бинарных признаков
        binary_encoder = ZeroBinaryEncoder(alpha=0.05)
        X_with_binary = binary_encoder.fit_transform(X_engineered, y_clean)
        
        # Шаг 4: Удаление неинформативных признаков
        uninform_remover = FeatureUninformRemove(threshold=0.95, verbose=False)
        X_informative = uninform_remover.fit_transform(X_with_binary)
        
        # Шаг 5: Удаление мультиколлинеарности
        collinearity_reducer = CollinearityReducer(verbose=False)
        X_reduced = collinearity_reducer.fit_transform(X_informative)
        
        # Проверяем, что все шаги выполнены успешно
        assert len(X_reduced) == len(X_clean)
        assert X_reduced.shape[1] <= X_with_binary.shape[1]
        assert 'Strength' not in X_reduced.columns  # Целевая переменная должна быть отдельно
        
        # Проверяем, что данные валидны
        assert not X_reduced.isnull().any().any()
        assert not np.any(np.isinf(X_reduced.values))


if __name__ == '__main__':
    # Запуск тестов
    pytest.main([__file__, '-v'])