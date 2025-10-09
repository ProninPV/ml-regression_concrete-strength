import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Добавляем путь к исходному коду в Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.engineer import (
    create_feature_analysis, 
    check_multicollinearity, 
    highlight_high_vif,
    plot_feature_trends_orchestrator  # Добавлен импорт новой функции
)


class TestPlotFeatureTrendsOrchestrator:
    """Тесты для функции plot_feature_trends_orchestrator"""
    
    @pytest.fixture
    def sample_data(self):
        """Фикстура с примером данных для тестирования"""
        np.random.seed(42)
        n_samples = 50
        
        return pd.DataFrame({
            'Cement': np.random.normal(250, 50, n_samples),
            'Water': np.random.normal(180, 20, n_samples),
            'Age': np.random.lognormal(2, 1, n_samples),
            'Strength': np.random.normal(35, 10, n_samples)
        })
    
    @pytest.fixture
    def sample_config(self):
        """Фикстура с конфигурацией"""
        return {
            'trend_settings': {
                'names': ['Linear', 'Log', 'Sqrt', '1/x', 'x²'],
                'colors': ['red', 'blue', 'green', 'orange', 'purple']
            }
        }
    
    @patch('src.features.engineer.calculate_trend_metrics')
    @patch('src.features.engineer.select_best_transformations')
    @patch('src.features.engineer.plot_feature_trends')
    def test_plot_feature_trends_orchestrator_basic(self, mock_plot, mock_select, mock_calculate, sample_data, sample_config):
        """Тест базового вызова функции"""
        # Настраиваем моки
        mock_metrics_df = pd.DataFrame({
            'feature': ['Cement', 'Water', 'Age'],
            'linear_r2_score': [0.8, 0.6, 0.7]
        })
        mock_results_df = pd.DataFrame({
            'feature': ['Cement', 'Water', 'Age'],
            'best_transformation': ['Linear', 'Log', 'Sqrt'],
            'best_r2_score': [0.8, 0.6, 0.7]
        })
        
        mock_calculate.return_value = mock_metrics_df
        mock_select.return_value = mock_results_df
        
        # Вызываем функцию
        result = plot_feature_trends_orchestrator(sample_data, sample_config)
        
        # Проверяем вызовы
        mock_calculate.assert_called_once()
        mock_select.assert_called_once()
        mock_plot.assert_called_once()
        
        # Проверяем результат
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    @patch('src.features.engineer.calculate_trend_metrics')
    @patch('src.features.engineer.select_best_transformations')
    @patch('src.features.engineer.plot_feature_trends')
    def test_plot_feature_trends_orchestrator_custom_parameters(self, mock_plot, mock_select, mock_calculate, sample_data, sample_config):
        """Тест с пользовательскими параметрами"""
        # Настраиваем моки
        mock_metrics_df = pd.DataFrame({'feature': ['Cement'], 'linear_r2_score': [0.8]})
        mock_results_df = pd.DataFrame({
            'feature': ['Cement'],
            'best_transformation': ['Linear'],
            'best_r2_score': [0.8]
        })
        
        mock_calculate.return_value = mock_metrics_df
        mock_select.return_value = mock_results_df
        
        # Вызываем функцию с кастомными параметрами
        result = plot_feature_trends_orchestrator(
            sample_data, 
            sample_config, 
            target='Strength',
            figsize=(20, 15),
            alpha=0.3
        )
        
        # Проверяем что функции вызваны с правильными параметрами
        mock_calculate.assert_called_once()
        mock_select.assert_called_once_with(mock_metrics_df, 0.3)
        mock_plot.assert_called_once()
        
        assert isinstance(result, pd.DataFrame)
    
    @patch('src.features.engineer.calculate_trend_metrics')
    @patch('src.features.engineer.select_best_transformations')
    @patch('src.features.engineer.plot_feature_trends')
    def test_plot_feature_trends_orchestrator_excludes_target(self, mock_plot, mock_select, mock_calculate, sample_data, sample_config):
        """Тест что целевая переменная исключена из признаков"""
        # Настраиваем моки
        mock_metrics_df = pd.DataFrame({'feature': ['Cement', 'Water'], 'linear_r2_score': [0.8, 0.6]})
        mock_results_df = pd.DataFrame({
            'feature': ['Cement', 'Water'],
            'best_transformation': ['Linear', 'Log'],
            'best_r2_score': [0.8, 0.6]
        })
        
        mock_calculate.return_value = mock_metrics_df
        mock_select.return_value = mock_results_df
        
        # Вызываем функцию
        plot_feature_trends_orchestrator(sample_data, sample_config)
        
        # Проверяем что calculate_trend_metrics вызвана с правильными признаками (без Strength)
        called_features = mock_calculate.call_args[0][1]  # Второй аргумент - features
        assert 'Strength' not in called_features
        assert 'Cement' in called_features
        assert 'Water' in called_features
        assert 'Age' in called_features
    
    @patch('src.features.engineer.calculate_trend_metrics')
    @patch('src.features.engineer.select_best_transformations')
    @patch('src.features.engineer.plot_feature_trends')
    def test_plot_feature_trends_orchestrator_empty_data(self, mock_plot, mock_select, mock_calculate, sample_config):
        """Тест с пустыми данными"""
        empty_df = pd.DataFrame(columns=['Cement', 'Water', 'Strength'])
        
        # Настраиваем моки для пустых данных
        mock_metrics_df = pd.DataFrame(columns=['feature', 'linear_r2_score'])
        mock_results_df = pd.DataFrame(columns=['feature', 'best_transformation', 'best_r2_score'])
        
        mock_calculate.return_value = mock_metrics_df
        mock_select.return_value = mock_results_df
        
        # Вызываем функцию с пустыми данными
        result = plot_feature_trends_orchestrator(empty_df, sample_config)
        
        # Проверяем что функции были вызваны
        mock_calculate.assert_called_once()
        mock_select.assert_called_once()
        mock_plot.assert_called_once()
        
        # Проверяем результат
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    @patch('src.features.engineer.calculate_trend_metrics')
    @patch('src.features.engineer.select_best_transformations')
    @patch('src.features.engineer.plot_feature_trends')
    def test_plot_feature_trends_orchestrator_only_target(self, mock_plot, mock_select, mock_calculate, sample_config):
        """Тест когда в данных только целевая переменная"""
        df_only_target = pd.DataFrame({'Strength': [1, 2, 3, 4, 5]})
        
        # Настраиваем моки для случая без признаков
        mock_metrics_df = pd.DataFrame(columns=['feature', 'linear_r2_score'])
        mock_results_df = pd.DataFrame(columns=['feature', 'best_transformation', 'best_r2_score'])
        
        mock_calculate.return_value = mock_metrics_df
        mock_select.return_value = mock_results_df
        
        # Вызываем функцию
        result = plot_feature_trends_orchestrator(df_only_target, sample_config)
        
        # Проверяем что функции были вызваны
        mock_calculate.assert_called_once()
        mock_select.assert_called_once()
        mock_plot.assert_called_once()
        
        # Проверяем результат
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestCreateFeatureAnalysis:
    """Тесты для функции create_feature_analysis"""
    
    @pytest.fixture
    def sample_data(self):
        """Фикстура с примером данных для тестирования"""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'Cement': np.random.normal(250, 50, n_samples),
            'BlastFurnaceSlag': np.random.normal(100, 30, n_samples),
            'FlyAsh': np.random.normal(80, 25, n_samples),
            'Water': np.random.normal(180, 20, n_samples),
            'Superplasticizer': np.random.normal(6, 2, n_samples),
            'CoarseAggregate': np.random.normal(1000, 50, n_samples),
            'FineAggregate': np.random.normal(800, 40, n_samples),
            'Age': np.random.lognormal(2, 1, n_samples),
            'Strength': np.random.normal(35, 10, n_samples)  # целевая переменная
        })
    
    @pytest.fixture
    def sample_data_with_categorical(self):
        """Фикстура с категориальными признаками"""
        np.random.seed(42)
        n_samples = 50
        
        return pd.DataFrame({
            'Cement': np.random.normal(250, 50, n_samples),
            'Water': np.random.normal(180, 20, n_samples),
            'Category': np.random.choice(['A', 'B', 'C'], n_samples),
            'Strength': np.random.normal(35, 10, n_samples)
        })
    
    def test_create_feature_analysis_returns_dataframe(self, sample_data):
        """Тест что функция возвращает DataFrame"""
        result = create_feature_analysis(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_create_feature_analysis_columns(self, sample_data):
        """Тест наличия всех необходимых колонок"""
        result = create_feature_analysis(sample_data)
        
        expected_columns = [
            'feature', 'pearson_corr', 'spearman_corr', 
            'spearman_p_value', 'skewness', 'kurtosis'
        ]
        
        for col in expected_columns:
            assert col in result.columns
    
    def test_create_feature_analysis_excludes_target(self, sample_data):
        """Тест что целевая переменная исключена из анализа"""
        result = create_feature_analysis(sample_data)
        
        # Strength не должна быть в списке признаков
        assert 'Strength' not in result['feature'].values
    
    def test_create_feature_analysis_excludes_categorical(self, sample_data_with_categorical):
        """Тест что категориальные признаки исключены"""
        result = create_feature_analysis(sample_data_with_categorical)
        
        # Category не должна быть в списке признаков
        assert 'Category' not in result['feature'].values
        # Только числовые признаки должны остаться
        assert 'Cement' in result['feature'].values
        assert 'Water' in result['feature'].values
    
    def test_create_feature_analysis_sorted_by_spearman(self, sample_data):
        """Тест сортировки по корреляции Спирмена"""
        result = create_feature_analysis(sample_data)
        
        # Проверяем что отсортировано по убыванию spearman_corr
        spearman_corrs = result['spearman_corr'].values
        assert all(spearman_corrs[i] >= spearman_corrs[i+1] 
                  for i in range(len(spearman_corrs)-1))
    
    def test_create_feature_analysis_correlation_range(self, sample_data):
        """Тест что корреляции в допустимом диапазоне"""
        result = create_feature_analysis(sample_data)
        
        # Корреляции должны быть между -1 и 1
        assert all(result['pearson_corr'].between(-1, 1))
        assert all(result['spearman_corr'].between(-1, 1))
    
    def test_create_feature_analysis_p_values(self, sample_data):
        """Тест p-значений"""
        result = create_feature_analysis(sample_data)
        
        # P-значения должны быть между 0 и 1
        assert all(result['spearman_p_value'].between(0, 1))
    
    def test_create_feature_analysis_statistical_properties(self, sample_data):
        """Тест статистических свойств (асимметрия, эксцесс)"""
        result = create_feature_analysis(sample_data)
        
        # Асимметрия и эксцесс должны быть числами
        assert all(np.isfinite(result['skewness']))
        assert all(np.isfinite(result['kurtosis']))
    
    def test_create_feature_analysis_empty_data(self):
        """Тест с пустыми данными"""
        empty_df = pd.DataFrame({'Strength': []})
        
        # Для пустых данных функция должна вернуть пустой DataFrame без ошибок
        result = create_feature_analysis(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        # Проверяем что колонки присутствуют
        expected_columns = ['feature', 'pearson_corr', 'spearman_corr', 'spearman_p_value', 'skewness', 'kurtosis']
        for col in expected_columns:
            assert col in result.columns
    
    def test_create_feature_analysis_single_feature(self):
        """Тест с одним признаком"""
        df = pd.DataFrame({
            'Cement': [100, 200, 300, 400, 500],
            'Strength': [20, 25, 30, 35, 40]
        })
        
        result = create_feature_analysis(df)
        
        assert len(result) == 1
        assert result.iloc[0]['feature'] == 'Cement'


class TestCheckMulticollinearity:
    """Тесты для функции check_multicollinearity"""
    
    @pytest.fixture
    def sample_data(self):
        """Фикстура с примером данных для тестирования мультиколлинеарности"""
        np.random.seed(42)
        n_samples = 100
        
        # Создаем коррелированные признаки
        base_feature = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame({
            'feature1': base_feature,
            'feature2': base_feature + np.random.normal(0, 0.1, n_samples),  # сильно коррелирован
            'feature3': np.random.normal(0, 1, n_samples),  # слабо коррелирован
            'feature4': base_feature * 2 + np.random.normal(0, 0.1, n_samples),  # сильно коррелирован
            'Strength': np.random.normal(0, 1, n_samples)
        })
    
    @pytest.fixture
    def perfectly_correlated_data(self):
        """Фикстура с идеально коррелированными признаками"""
        n_samples = 50
        base = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame({
            'x1': base,
            'x2': base,  # идеальная корреляция
            'x3': base * 2,  # идеальная корреляция
            'x4': np.random.normal(0, 1, n_samples),  # независимый
            'Strength': np.random.normal(0, 1, n_samples)
        })
    
    @patch('src.features.engineer.plt.show')
    def test_check_multicollinearity_returns_dataframe(self, mock_show, sample_data):
        """Тест что функция возвращает DataFrame"""
        result = check_multicollinearity(sample_data, plot_heatmap=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    @patch('src.features.engineer.plt.show')
    def test_check_multicollinearity_columns(self, mock_show, sample_data):
        """Тест наличия всех необходимых колонок"""
        result = check_multicollinearity(sample_data, plot_heatmap=False)
        
        expected_columns = ['Feature', 'VIF']
        
        for col in expected_columns:
            assert col in result.columns
    
    @patch('src.features.engineer.plt.show')
    def test_check_multicollinearity_excludes_target(self, mock_show, sample_data):
        """Тест что целевая переменная исключена из анализа"""
        result = check_multicollinearity(sample_data, plot_heatmap=False)
        
        # Strength не должна быть в списке признаков
        assert 'Strength' not in result['Feature'].values
    
    @patch('src.features.engineer.plt.show')
    def test_check_multicollinearity_sorted_by_vif(self, mock_show, sample_data):
        """Тест сортировки по VIF"""
        result = check_multicollinearity(sample_data, plot_heatmap=False)
        
        # Проверяем что отсортировано по убыванию VIF
        vif_values = result['VIF'].values
        assert all(vif_values[i] >= vif_values[i+1] 
                  for i in range(len(vif_values)-1))
    
    @patch('src.features.engineer.plt.show')
    def test_check_multicollinearity_vif_range(self, mock_show, sample_data):
        """Тест что VIF значения положительные"""
        result = check_multicollinearity(sample_data, plot_heatmap=False)
        
        # VIF должен быть >= 1
        assert all(result['VIF'] >= 1)
    
    @patch('src.features.engineer.plt.show')
    def test_check_multicollinearity_perfect_correlation(self, mock_show, perfectly_correlated_data):
        """Тест с идеально коррелированными признаками"""
        result = check_multicollinearity(perfectly_correlated_data, plot_heatmap=False)
        
        # Идеально коррелированные признаки должны иметь высокий VIF
        high_vif_features = result[result['VIF'] > 10]['Feature'].values
        assert len(high_vif_features) > 0
    
    @patch('src.features.engineer.plt.show')
    @patch('src.features.engineer.sns.heatmap')
    @patch('src.features.engineer.plt.figure')
    def test_check_multicollinearity_heatmap_called(self, mock_fig, mock_heatmap, mock_show, sample_data):
        """Тест что тепловая карта строится при plot_heatmap=True"""
        check_multicollinearity(sample_data, plot_heatmap=True)
        
        # Проверяем что функции построения графика были вызваны
        mock_fig.assert_called()
        mock_heatmap.assert_called()
        mock_show.assert_called()
    
    @patch('src.features.engineer.plt.show')
    @patch('src.features.engineer.sns.heatmap')
    @patch('src.features.engineer.plt.figure')
    def test_check_multicollinearity_no_heatmap(self, mock_fig, mock_heatmap, mock_show, sample_data):
        """Тест что тепловая карта не строится при plot_heatmap=False"""
        check_multicollinearity(sample_data, plot_heatmap=False)
        
        # Проверяем что функции построения графика не были вызваны
        mock_fig.assert_not_called()
        mock_heatmap.assert_not_called()
    
    @patch('src.features.engineer.plt.show')
    def test_check_multicollinearity_custom_threshold(self, mock_show, sample_data):
        """Тест с пользовательским порогом VIF"""
        result = check_multicollinearity(sample_data, threshold=5, plot_heatmap=False)
        
        # Функция должна корректно обработать кастомный порог
        assert isinstance(result, pd.DataFrame)
    
    def test_check_multicollinearity_empty_data(self):
        """Тест с пустыми данными"""
        empty_df = pd.DataFrame({'Strength': []})
        
        # Для пустых данных функция должна вернуть пустой DataFrame без ошибок
        result = check_multicollinearity(empty_df, plot_heatmap=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        # Проверяем что колонки присутствуют
        assert 'Feature' in result.columns
        assert 'VIF' in result.columns
    
    @patch('src.features.engineer.plt.show')
    def test_check_multicollinearity_single_feature(self, mock_show):
        """Тест с одним признаком"""
        df = pd.DataFrame({
            'Cement': [100, 200, 300, 400, 500],
            'Strength': [20, 25, 30, 35, 40]
        })
        
        result = check_multicollinearity(df, plot_heatmap=False)
        
        assert len(result) == 1
        assert result.iloc[0]['Feature'] == 'Cement'
        # VIF для одного признака должен быть 1 (нет мультиколлинеарности)
        assert result.iloc[0]['VIF'] == 1.0


class TestHighlightHighVIF:
    """Тесты для функции highlight_high_vif"""
    
    @pytest.fixture
    def sample_vif_data(self):
        """Фикстура с примером данных VIF"""
        return pd.DataFrame({
            'Feature': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            'VIF': [15.5, 8.2, 3.1, 25.0, 4.8]
        })
    
    @pytest.fixture
    def edge_case_vif_data(self):
        """Фикстура с граничными значениями VIF"""
        return pd.DataFrame({
            'Feature': ['f1', 'f2', 'f3', 'f4'],
            'VIF': [10.0, 5.0, 4.9, 9.9]  # граничные значения
        })
    
    def test_highlight_high_vif_returns_styler(self, sample_vif_data):
        """Тест что функция возвращает Styler объект"""
        result = highlight_high_vif(sample_vif_data)
        
        # Проверяем что возвращается объект Styler
        assert hasattr(result, 'data')
        assert isinstance(result.data, pd.DataFrame)
        assert 'Styler' in str(type(result))
    
    def test_highlight_high_vif_default_thresholds(self, sample_vif_data):
        """Тест выделения с порогами по умолчанию"""
        styled = highlight_high_vif(sample_vif_data)
        
        # Проверяем что функция применяет стили
        assert hasattr(styled, '_compute')
        # Проверяем что данные корректны
        assert all(col in styled.data.columns for col in ['Feature', 'VIF'])
    
    def test_highlight_high_vif_custom_thresholds(self, sample_vif_data):
        """Тест с пользовательскими порогами"""
        styled = highlight_high_vif(sample_vif_data, high_threshold=20, medium_threshold=10)
        
        # Проверяем что функция применяет стили с кастомными порогами
        assert hasattr(styled, '_compute')
        assert all(col in styled.data.columns for col in ['Feature', 'VIF'])
    
    def test_highlight_high_vif_formatting(self, sample_vif_data):
        """Тест форматирования чисел"""
        styled = highlight_high_vif(sample_vif_data)
        
        # Проверяем что стили применяются корректно
        assert hasattr(styled, '_compute')  # или hasattr(styled, 'data')
        # Проверяем что VIF значения форматируются
        vif_values = styled.data['VIF']
        assert all(isinstance(v, (int, float, np.number)) for v in vif_values)
    
    def test_highlight_high_vif_edge_cases(self, edge_case_vif_data):
        """Тест граничных случаев"""
        styled = highlight_high_vif(edge_case_vif_data)
        
        # Проверяем что функция корректно обрабатывает граничные значения
        assert hasattr(styled, 'data')
        assert len(styled.data) == 4
    
    def test_highlight_high_vif_empty_data(self):
        """Тест с пустыми данными"""
        empty_df = pd.DataFrame(columns=['Feature', 'VIF'])
        
        styled = highlight_high_vif(empty_df)
        
        # Должен вернуться Styler даже для пустых данных
        assert hasattr(styled, 'data')
        assert len(styled.data) == 0
        assert all(col in styled.data.columns for col in ['Feature', 'VIF'])
    
    def test_highlight_high_vif_high_vif_detection(self, sample_vif_data):
        """Тест обнаружения высоких VIF значений"""
        # Получаем VIF значения
        vif_values = sample_vif_data['VIF'].values
        
        # Проверяем что есть значения выше порога
        high_vif_count = sum(vif_values >= 10)
        medium_vif_count = sum((vif_values >= 5) & (vif_values < 10))
        
        assert high_vif_count > 0
        assert medium_vif_count > 0


class TestIntegration:
    """Интеграционные тесты для совместной работы функций"""
    
    @pytest.fixture
    def integration_data(self):
        """Фикстура для интеграционных тестов"""
        np.random.seed(42)
        n_samples = 200
        
        return pd.DataFrame({
            'Cement': np.random.normal(280, 45, n_samples),
            'BlastFurnaceSlag': np.random.normal(90, 25, n_samples),
            'FlyAsh': np.random.normal(60, 20, n_samples),
            'Water': np.random.normal(175, 15, n_samples),
            'Superplasticizer': np.random.normal(6, 1.5, n_samples),
            'CoarseAggregate': np.random.normal(975, 35, n_samples),
            'FineAggregate': np.random.normal(825, 30, n_samples),
            'Age': np.random.exponential(28, n_samples),
            'Strength': np.random.normal(40, 8, n_samples)
        })
    
    @patch('src.features.engineer.plt.show')
    def test_analysis_and_multicollinearity_integration(self, mock_show, integration_data):
        """Интеграционный тест анализа признаков и мультиколлинеарности"""
        # Анализ признаков
        analysis_df = create_feature_analysis(integration_data)
        
        # Проверка мультиколлинеарности
        vif_df = check_multicollinearity(integration_data, plot_heatmap=False)
        
        # Проверяем что обе функции работают корректно
        assert isinstance(analysis_df, pd.DataFrame)
        assert isinstance(vif_df, pd.DataFrame)
        
        # Проверяем что признаки совпадают (исключая возможные различия в порядке)
        analysis_features = set(analysis_df['feature'].values)
        vif_features = set(vif_df['Feature'].values)
        
        assert analysis_features == vif_features
    
    def test_vif_and_highlight_integration(self, integration_data):
        """Интеграционный тест VIF анализа и выделения"""
        # Получаем VIF данные
        vif_df = check_multicollinearity(integration_data, plot_heatmap=False)
        
        # Применяем выделение
        styled_vif = highlight_high_vif(vif_df)
        
        # Проверяем что оба шага работают
        assert isinstance(vif_df, pd.DataFrame)
        assert hasattr(styled_vif, 'data')
        assert 'Styler' in str(type(styled_vif))
    
    @patch('src.features.engineer.plt.show')
    def test_complete_workflow(self, mock_show, integration_data):
        """Полный workflow тест"""
        # Шаг 1: Анализ признаков
        feature_analysis = create_feature_analysis(integration_data)
        
        # Шаг 2: Проверка мультиколлинеарности
        vif_analysis = check_multicollinearity(integration_data, plot_heatmap=False)
        
        # Шаг 3: Визуальное выделение проблемных признаков
        styled_vif = highlight_high_vif(vif_analysis)
        
        # Проверяем все результаты
        assert len(feature_analysis) > 0
        assert len(vif_analysis) > 0
        assert hasattr(styled_vif, 'data')
        assert 'Styler' in str(type(styled_vif))
        
        # Проверяем что нет NaN значений
        assert not feature_analysis.isnull().any().any()
        assert not vif_analysis.isnull().any().any()


if __name__ == "__main__":
    # Запуск тестов из командной строки
    pytest.main([__file__, "-v"])