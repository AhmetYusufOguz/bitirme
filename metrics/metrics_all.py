"""
Performans metrikleri modülü
IGD, Hyper-Volume, QM, SM metriklerini içerir
"""

import numpy as np
from typing import List, Tuple
from core.solution import Solution, ParetoFront


# ============================================================================
# IGD.PY - Inverted Generational Distance
# ============================================================================

class IGDMetric:
    """
    Inverted Generational Distance (IGD)
    Makaledeki Equation (25)
    
    Elde edilen Pareto frontunun optimal Pareto frontuna ne kadar yakın olduğunu ölçer.
    Daha düşük değer daha iyidir.
    """
    
    @staticmethod
    def calculate(obtained_front: ParetoFront, 
                  reference_front: ParetoFront) -> float:
        """
        IGD hesapla
        
        Args:
            obtained_front: Algoritma tarafından bulunan front
            reference_front: Referans (optimal/yaklaşık optimal) front
            
        Returns:
            IGD değeri
        """
        obtained_objectives = obtained_front.get_objectives_array()
        reference_objectives = reference_front.get_objectives_array()
        
        if len(reference_objectives) == 0:
            return float('inf')
        
        if len(obtained_objectives) == 0:
            return float('inf')
        
        # Her referans nokta için en yakın elde edilen noktaya mesafe
        total_distance = 0.0
        
        for ref_point in reference_objectives:
            min_distance = float('inf')
            
            for obt_point in obtained_objectives:
                # Euclidean mesafe
                distance = np.sqrt(np.sum((ref_point - obt_point) ** 2))
                if distance < min_distance:
                    min_distance = distance
            
            total_distance += min_distance
        
        # Ortalama mesafe
        igd = total_distance / len(reference_objectives)
        
        return igd
    
    @staticmethod
    def create_reference_front(all_fronts: List[ParetoFront]) -> ParetoFront:
        """
        Birden fazla frontu birleştirerek referans front oluştur
        Tüm algoritmaların non-dominated çözümlerini içerir
        """
        combined_front = ParetoFront()
        
        for front in all_fronts:
            for solution in front.get_solutions():
                combined_front.add(solution)
        
        return combined_front


# ============================================================================
# HYPERVOLUME.PY - Hyper-Volume Metric
# ============================================================================

class HyperVolumeMetric:
    """
    Hyper-Volume (HV) Metric
    Makaledeki Equation (26)
    
    Pareto frontunun kapladığı hacmi ölçer.
    Daha yüksek değer daha iyidir.
    """
    
    @staticmethod
    def calculate(pareto_front: ParetoFront, 
                  reference_point: Tuple[float, float]) -> float:
        """
        Hyper-volume hesapla
        
        Args:
            pareto_front: Değerlendirilecek Pareto frontu
            reference_point: Referans nokta (r1, r2) - tüm çözümleri domine eden
            
        Returns:
            Hyper-volume değeri
        """
        objectives = pareto_front.get_objectives_array()
        
        if len(objectives) == 0:
            return 0.0
        
        # 2D için basitleştirilmiş hesaplama
        # Çözümleri f1'e göre sırala
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objectives = objectives[sorted_indices]
        
        hypervolume = 0.0
        
        # Her çözüm için dikdörtgen alan hesapla
        for i in range(len(sorted_objectives)):
            f1, f2 = sorted_objectives[i]
            
            # Genişlik: bir sonraki noktaya kadar (veya referans noktaya)
            if i < len(sorted_objectives) - 1:
                width = sorted_objectives[i + 1][0] - f1
            else:
                width = reference_point[0] - f1
            
            # Yükseklik: referans noktadan bu noktaya
            height = reference_point[1] - f2
            
            # Alan
            if width > 0 and height > 0:
                hypervolume += width * height
        
        return hypervolume
    
    @staticmethod
    def get_reference_point(all_fronts: List[ParetoFront], 
                           margin: float = 1.1) -> Tuple[float, float]:
        """
        Tüm frontlar için uygun referans nokta oluştur
        
        Args:
            all_fronts: Tüm Pareto frontları
            margin: Maksimum değerlere eklenecek marj (çarpan)
            
        Returns:
            Referans nokta (r1, r2)
        """
        max_f1 = 0.0
        max_f2 = 0.0
        
        for front in all_fronts:
            objectives = front.get_objectives_array()
            if len(objectives) > 0:
                max_f1 = max(max_f1, np.max(objectives[:, 0]))
                max_f2 = max(max_f2, np.max(objectives[:, 1]))
        
        # Marj ekle
        ref_f1 = max_f1 * margin
        ref_f2 = max_f2 * margin
        
        return (ref_f1, ref_f2)


# ============================================================================
# OTHERS.PY - QM ve SM Metrikleri
# ============================================================================

class QMMetric:
    """
    Quantity Metric (QM)
    Makaledeki Section 4.2.2
    
    Bir algoritmanın tüm algoritmalar arasında bulduğu 
    non-dominated çözümlerin oranı.
    Daha yüksek değer daha iyidir.
    """
    
    @staticmethod
    def calculate(algorithm_front: ParetoFront, 
                  all_fronts: List[ParetoFront]) -> float:
        """
        QM hesapla
        
        Args:
            algorithm_front: Değerlendirilen algoritmanın frontu
            all_fronts: Tüm algoritmaların frontları
            
        Returns:
            QM değeri (0-1 arası)
        """
        # Birleşik referans front oluştur
        combined_front = IGDMetric.create_reference_front(all_fronts)
        
        total_solutions = combined_front.size()
        if total_solutions == 0:
            return 0.0
        
        # Bu algoritmanın birleşik fronttaki çözüm sayısı
        algorithm_solutions = algorithm_front.get_solutions()
        algorithm_objectives = algorithm_front.get_objectives_array()
        
        count = 0
        combined_objectives = combined_front.get_objectives_array()
        
        # Bu algoritmanın her çözümü için
        for algo_obj in algorithm_objectives:
            # Birleşik frontta var mı?
            for comb_obj in combined_objectives:
                if np.allclose(algo_obj, comb_obj, rtol=1e-5):
                    count += 1
                    break
        
        qm = count / total_solutions
        return qm


class SMMetric:
    """
    Spacing Metric (SM)
    Makaledeki Equation (24)
    
    Pareto frontundaki çözümlerin dağılımının düzgünlüğünü ölçer.
    Daha düşük değer daha iyidir (daha düzgün dağılım).
    """
    
    @staticmethod
    def calculate(pareto_front: ParetoFront) -> float:
        """
        SM hesapla
        
        Args:
            pareto_front: Değerlendirilecek Pareto frontu
            
        Returns:
            SM değeri
        """
        objectives = pareto_front.get_objectives_array()
        
        if len(objectives) <= 1:
            return 0.0
        
        n = len(objectives)
        distances = []
        
        # Her çözüm için en yakın komşusuna mesafe
        for i in range(n):
            min_distance = float('inf')
            
            for j in range(n):
                if i != j:
                    # Manhattan distance (makaledeki gibi)
                    distance = np.sum(np.abs(objectives[i] - objectives[j]))
                    if distance < min_distance:
                        min_distance = distance
            
            distances.append(min_distance)
        
        # Ortalama mesafe
        d_mean = np.mean(distances)
        
        # Standart sapma
        spacing = np.sqrt(np.sum((distances - d_mean) ** 2) / (n - 1))
        
        return spacing


class MIDMetric:
    """
    Mean Ideal Distance (MID)
    Makaledeki Section 4.2.1
    
    İdeal noktaya (0, 0) ortalama mesafe.
    Daha düşük değer daha iyidir.
    """
    
    @staticmethod
    def calculate(pareto_front: ParetoFront, 
                  ideal_point: Tuple[float, float] = (0.0, 0.0)) -> float:
        """
        MID hesapla
        
        Args:
            pareto_front: Değerlendirilecek Pareto frontu
            ideal_point: İdeal nokta (varsayılan: (0, 0))
            
        Returns:
            MID değeri
        """
        objectives = pareto_front.get_objectives_array()
        
        if len(objectives) == 0:
            return float('inf')
        
        ideal = np.array(ideal_point)
        
        # Her çözümün ideal noktaya mesafesi
        distances = []
        for obj in objectives:
            distance = np.sqrt(np.sum((obj - ideal) ** 2))
            distances.append(distance)
        
        # Ortalama mesafe
        mid = np.mean(distances)
        
        return mid


# ============================================================================
# COMPARISON - Algoritma Karşılaştırma Sınıfı
# ============================================================================

class AlgorithmComparison:
    """
    Birden fazla algoritmayı karşılaştırmak için yardımcı sınıf
    """
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, algorithm_name: str, pareto_front: ParetoFront):
        """Algoritma sonucu ekle"""
        self.results[algorithm_name] = pareto_front
    
    def compare_all(self) -> dict:
        """
        Tüm algoritmaları karşılaştır
        
        Returns:
            Her metrik için algoritma karşılaştırması
        """
        if len(self.results) == 0:
            return {}
        
        all_fronts = list(self.results.values())
        
        # Referans front ve nokta
        reference_front = IGDMetric.create_reference_front(all_fronts)
        reference_point = HyperVolumeMetric.get_reference_point(all_fronts)
        
        comparison = {}
        
        for algo_name, front in self.results.items():
            comparison[algo_name] = {
                'IGD': IGDMetric.calculate(front, reference_front),
                'HV': HyperVolumeMetric.calculate(front, reference_point),
                'QM': QMMetric.calculate(front, all_fronts),
                'SM': SMMetric.calculate(front),
                'MID': MIDMetric.calculate(front),
                'Size': front.size()
            }
        
        return comparison
    
    def print_comparison(self):
        """Karşılaştırma sonuçlarını yazdır"""
        comparison = self.compare_all()
        
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON RESULTS")
        print("="*80)
        
        # Header
        print(f"{'Algorithm':<15} {'IGD':<12} {'HV':<12} {'QM':<8} {'SM':<10} {'MID':<12} {'Size':<6}")
        print("-"*80)
        
        # Her algoritma için
        for algo_name, metrics in comparison.items():
            print(f"{algo_name:<15} "
                  f"{metrics['IGD']:<12.4f} "
                  f"{metrics['HV']:<12.2f} "
                  f"{metrics['QM']:<8.4f} "
                  f"{metrics['SM']:<10.4f} "
                  f"{metrics['MID']:<12.2f} "
                  f"{metrics['Size']:<6}")
        
        print("="*80)
        print("Lower is better: IGD, SM, MID")
        print("Higher is better: HV, QM, Size")
        print("="*80 + "\n")