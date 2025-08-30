# ecg_predictor.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ecg_preprocessing import ECGPreprocessor

class ECGPredictor:
    def __init__(self, model_path='best_ecg_model_updated.h5', apply_preprocessing=True):
        """Initialize the ECG predictor"""
        self.model = tf.keras.models.load_model(model_path)
        self.category_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown/Noise', 'AFib', 'ST changes']
        self.apply_preprocessing = apply_preprocessing
        self.preprocessor = ECGPreprocessor(sampling_rate=250)
        
        # Define severity levels and recommendations for each arrhythmia type
        self.severity_levels = {
            'Normal': 'low',
            'Supraventricular': 'medium',
            'Ventricular': 'high',
            'Fusion': 'medium',
            'Unknown/Noise': 'low',
            'AFib': 'high',
            'ST changes': 'critical'
        }
        
        self.recommendations = {
            'low': {
                'action': 'Stay home and monitor',
                'description': 'Continue normal activities but monitor for any changes in symptoms.',
                'urgency': 'Low',
                'color': 'green'
            },
            'medium': {
                'action': 'Schedule doctor appointment',
                'description': 'Contact your healthcare provider within 24-48 hours for evaluation.',
                'urgency': 'Medium',
                'color': 'orange'
            },
            'high': {
                'action': 'Seek immediate medical attention',
                'description': 'Go to urgent care or emergency room for immediate evaluation.',
                'urgency': 'High',
                'color': 'red'
            },
            'critical': {
                'action': 'Go to hospital immediately',
                'description': 'Call emergency services (911) or go to nearest emergency room immediately.',
                'urgency': 'Critical',
                'color': 'darkred'
            }
        }
        
        print("✅ ECG Predictor initialized successfully!")
    
    def get_recommendation(self, arrhythmia_type, confidence):
        """Get recommendation based on arrhythmia type and confidence"""
        severity = self.severity_levels.get(arrhythmia_type, 'low')
        
        # Adjust severity if confidence is low
        if confidence < 0.5:
            if severity == 'critical':
                severity = 'high'
            elif severity == 'high':
                severity = 'medium'
        
        recommendation = self.recommendations[severity].copy()
        
        arrhythmia_notes = {
            'Normal': 'Continue regular monitoring and healthy lifestyle.',
            'Supraventricular': 'May require medication or ablation therapy.',
            'Ventricular': 'Can be life-threatening, requires immediate attention.',
            'Fusion': 'May indicate conduction system disease.',
            'Unknown/Noise': 'Consider retesting with better signal quality.',
            'AFib': 'Increases stroke risk, requires anticoagulation consideration.',
            'ST changes': 'May indicate myocardial ischemia or infarction.'
        }
        
        recommendation['arrhythmia_note'] = arrhythmia_notes.get(arrhythmia_type, '')
        recommendation['severity_level'] = severity
        recommendation['confidence'] = confidence
        
        return recommendation
    
    def predict(self, ecg_data):
        """Make prediction on ECG data"""
        if len(ecg_data) != 2500:
            raise ValueError(f"ECG data must be 2500 samples (10 seconds at 250 Hz), got {len(ecg_data)}")
        
        if self.apply_preprocessing:
            ecg_data = self.preprocessor.complete_preprocessing(ecg_data, visualize=False)
        
        input_data = ecg_data.reshape(1, 2500, 1)
        prediction = self.model.predict(input_data, verbose=0)
        
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        arrhythmia_type = self.category_names[predicted_class]
        
        recommendation = self.get_recommendation(arrhythmia_type, confidence)
        
        return {
            'class': arrhythmia_type,
            'confidence': confidence,
            'probabilities': prediction[0].tolist(),
            'recommendation': recommendation
        }
    
    def predict_with_visualization(self, ecg_data, result_override=None, save_plot=True):
        """Make prediction and visualize the result (with optional override for demo)"""
        result = self.predict(ecg_data)
        
        if result_override:
            fake_probs = [0.025] * len(self.category_names)
            idx = self.category_names.index(result_override['class'])
            fake_probs[idx] = result_override['confidence']
            
            result['class'] = result_override['class']
            result['confidence'] = result_override['confidence']
            result['probabilities'] = fake_probs
            result['recommendation'] = result_override['recommendation']
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        
        # Plot ECG signal
        ax1.plot(ecg_data)
        ax1.set_title('ECG Signal (10-second window)')
        ax1.set_xlabel('Samples (250 Hz)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # Plot probabilities
        categories = self.category_names
        probabilities = result['probabilities']
        
        bars = ax2.bar(categories, probabilities)
        ax2.set_title('Prediction Probabilities')
        ax2.set_xlabel('Arrhythmia Type')
        ax2.set_ylabel('Probability')
        ax2.tick_params(axis='x', rotation=45)
        
        predicted_idx = self.category_names.index(result['class'])
        bars[predicted_idx].set_color('red')
        
        ax2.text(0.02, 0.95, f"Predicted: {result['class']}\nConfidence: {result['confidence']:.3f}", 
                 transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Recommendation
        rec = result['recommendation']
        urgency_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'darkred'}
        
        ax3.text(0.05, 0.7, f"RECOMMENDATION", fontsize=16, fontweight='bold', 
                 color=urgency_colors[rec['urgency']])
        ax3.text(0.05, 0.5, f"Action: {rec['action']}", fontsize=14, fontweight='bold')
        ax3.text(0.05, 0.3, f"Urgency: {rec['urgency']}", fontsize=12, 
                 color=urgency_colors[rec['urgency']], fontweight='bold')
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        if save_plot:
            plt.savefig('ecg_prediction_result.png', dpi=300, bbox_inches='tight')
            print("📊 Visualization saved as 'ecg_prediction_result.png'")
        plt.show()
        
        return result

def demo():
    """Demo function to show how to use the predictor"""
    print("🏥 ECG Arrhythmia Predictor Demo (ecg_predictor.py)")
    print("=" * 50)
    
    # Initialize predictor (apply preprocessing by default)
    predictor = ECGPredictor()
    
    # Randomly select an arrhythmia type to simulate 
    arrhythmia_types = ['Supraventricular', 'Ventricular', 'Fusion', 'AFib', 'ST changes']
    selected_type = np.random.choice(arrhythmia_types)
    print(f"🎲 Simulating: {selected_type} ECG pattern")
    
    # Simulate a demo ECG signal 
    t = np.linspace(0, 10, 2500)
    ecg_demo = np.zeros(2500)
    ecg_demo += 0.1 * np.random.randn(2500)
    ecg_demo += 0.05 * np.sin(2 * np.pi * 0.1 * t)
    
    print("✅ Demo ECG data created!")
    
    confidence_f = round(np.random.uniform(0.80, 0.97), 3)
    
    result = {
        'class': selected_type,
        'confidence': confidence_f,
        'probabilities': [
            confidence_f if cat == selected_type else 
            round((1 - confidence_f) / (len(predictor.category_names) - 1), 3)
            for cat in predictor.category_names
        ],
        'recommendation': predictor.get_recommendation(selected_type, confidence_f)
    }
    
    # ---- Visualization ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot ECG signal
    ax1.plot(ecg_demo)
    ax1.set_title(f'ECG Signal (Simulated {selected_type})')
    ax1.set_xlabel('Samples (250 Hz)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Plot probabilities 
    categories = predictor.category_names
    bars = ax2.bar(categories, result['probabilities'])
    predicted_idx = categories.index(result['class'])
    bars[predicted_idx].set_color('red')
    ax2.set_title('Prediction Probabilities (Demo Override)')
    ax2.set_xlabel('Arrhythmia Type')
    ax2.set_ylabel('Probability')
    ax2.tick_params(axis='x', rotation=45)
    ax2.text(0.02, 0.95, f"Predicted: {result['class']}\nConfidence: {result['confidence']:.3f}",
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Recommendation box
    rec = result['recommendation']
    urgency_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'darkred'}
    ax3.text(0.05, 0.7, "RECOMMENDATION", fontsize=16, fontweight='bold',
             color=urgency_colors[rec['urgency']])
    ax3.text(0.05, 0.5, f"Action: {rec['action']}", fontsize=14, fontweight='bold')
    ax3.text(0.05, 0.3, f"Urgency: {rec['urgency']}", fontsize=12,
             color=urgency_colors[rec['urgency']], fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Print results to console
    print(f"\n📊 Prediction Result (Demo Override):")
    print(f"   Arrhythmia Type: {result['class']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print("   All probabilities:")
    for cat, prob in zip(predictor.category_names, result['probabilities']):
        print(f"     {cat}: {prob:.3f}")
    
    print(f"\n🏥 RECOMMENDATION:")
    print(f"   Action: {rec['action']}")
    print(f"   Urgency: {rec['urgency']}")
    print(f"   Description: {rec['description']}")
    print(f"   Note: {rec['arrhythmia_note']}")

if __name__ == "__main__":
    demo()
