import json
import os
from collections import defaultdict

def analyze_game_results(json_files, output_file="neo_analysis.json"):
    """
    Analyze game results from multiple JSON files and count how many times
    NEO was chosen for each image index.
    
    Args:
        json_files: List of paths to JSON files from game exports
        output_file: Path to save the ordered results
    """
    
    # Dictionary to track NEO choices per image
    # Key: image index (e.g., "img_00000")
    # Value: count of times NEO was chosen
    neo_choices = defaultdict(int)
    
    # Also track total times each image appeared for context
    image_appearances = defaultdict(int)
    
    # Process each JSON file
    for json_file in json_files:
        print(f"Processing: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Go through all choices in this file
            for choice in data.get('choices', []):
                index = choice.get('index')
                actual_type = choice.get('actual_type')
                
                # Create image name with padded index
                image_name = f"img_{str(index).zfill(5)}"
                
                # Track appearance
                image_appearances[image_name] += 1
                
                # If NEO was chosen, increment count
                if actual_type == 'neo':
                    neo_choices[image_name] += 1
        
        except FileNotFoundError:
            print(f"  ERROR: File not found - {json_file}")
        except json.JSONDecodeError:
            print(f"  ERROR: Invalid JSON format - {json_file}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Create ordered results
    results = []
    for image_name in sorted(neo_choices.keys()):
        results.append({
            'image': image_name,
            'neo_chosen_count': neo_choices[image_name],
            'total_appearances': image_appearances[image_name],
            'neo_chosen_percentage': round((neo_choices[image_name] / image_appearances[image_name] * 100), 2) if image_appearances[image_name] > 0 else 0
        })
    
    # Sort by neo_chosen_count (descending - most chosen first)
    results.sort(key=lambda x: x['neo_chosen_count'], reverse=True)
    
    # Create output data
    output_data = {
        'total_json_files_processed': len(json_files),
        'total_images_analyzed': len(results),
        'ordered_by': 'neo_chosen_count (descending)',
        'results': results
    }
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Total images analyzed: {len(results)}")
    
    # Print top 10
    print("\nTop 10 images where NEO was chosen most:")
    for i, result in enumerate(results[:10], 1):
        print(f"  {i}. {result['image']}: NEO chosen {result['neo_chosen_count']} times ({result['neo_chosen_percentage']}%)")
    
    return output_data


if __name__ == "__main__":
    # List your JSON files here
    json_files = [
        "qwen_run0.json",
        "qwen_run1.json",
        "qwen_run2.json",
        "qwen_run3.json",
        "qwen_run4.json"
    ]
    
    # You can also use a directory to automatically find all JSON files
    # Uncomment the following lines if all your JSON files are in a specific folder:
    
    # import glob
    # json_files = glob.glob("path/to/your/folder/*.json")
    
    # Run the analysis
    analyze_game_results(json_files, output_file="neo_analysis_ordered.json")