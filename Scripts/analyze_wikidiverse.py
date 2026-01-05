
#############################################
#### 1 Examine the Test Set Structure #######
#############################################

# import json
# from collections import defaultdict
# import re

# def load_test_data(test_file="test.json"):
#     """Load and examine the test data structure"""
#     with open(test_file, 'r') as f:
#         data = json.load(f)
    
#     print(f"Total test instances: {len(data)}")
    
#     # Examine first few items to understand structure
#     print("\nSample instance structure:")
#     for i, item in enumerate(data[:2]):
#         print(f"\nInstance {i}:")
#         print(f"  Caption: {item[0][:100]}...")
#         print(f"  Image URL: {item[1]}")
#         print(f"  Topic: {item[2]}")
#         print(f"  Entity annotations: {len(item[3])} entities")
#         for entity in item[3]:
#             print(f"    - {entity[0]} ({entity[1]}) -> {entity[4]}")
    
#     return data

# # Load and examine
# test_data = load_test_data()

#####################################
##### 2 Ambiguous Entity Pairs ######
#####################################

# def find_ambiguous_entity_pairs(test_data):
#     """
#     Find 'tricky' pairs where the same word form appears with different meanings
#     Returns: Dict of {entity_name: list_of_different_entity_urls}
#     """
    
#     # Track entities by their surface form
#     entity_form_to_urls = defaultdict(set)
#     entity_form_to_instances = defaultdict(list)
    
#     for idx, instance in enumerate(test_data):
#         caption = instance[0]
#         image_url = instance[1]
#         entities = instance[3]  # List of entity annotations
        
#         for entity in entities:
#             entity_name = entity[0]  # Surface form (e.g., "Apple")
#             entity_type = entity[1]  # Entity type
#             entity_url = entity[4]   # Wikipedia URL
            
#             # Store this occurrence
#             entity_form_to_urls[entity_name.lower()].add(entity_url)
#             entity_form_to_instances[entity_name.lower()].append({
#                 'instance_idx': idx,
#                 'entity_name': entity_name,
#                 'entity_url': entity_url,
#                 'entity_type': entity_type,
#                 'caption': caption,
#                 'image_url': image_url,
#                 'full_entity_data': entity
#             })
    
#     # Find ambiguous entities (same form, different URLs)
#     ambiguous_entities = {}
#     for entity_form, urls in entity_form_to_urls.items():
#         if len(urls) > 1:
#             ambiguous_entities[entity_form] = {
#                 'urls': list(urls),
#                 'instances': entity_form_to_instances[entity_form]
#             }
    
#     print(f"\n{'='*80}")
#     print("AMBIGUOUS ENTITY ANALYSIS")
#     print('='*80)
#     print(f"Total unique entity forms: {len(entity_form_to_urls)}")
#     print(f"Ambiguous entity forms: {len(ambiguous_entities)}")
#     print(f"Percentage ambiguous: {len(ambiguous_entities)/len(entity_form_to_urls)*100:.1f}%")
    
#     return ambiguous_entities, entity_form_to_instances

# # Find ambiguous pairs
# ambiguous_entities, entity_instances = find_ambiguous_entity_pairs(test_data)

#########################################################
#####Ambiguous Entity Pairs#############################
#######################################################

# def find_ambiguous_entity_pairs(test_data):
#     """
#     Find 'tricky' pairs where the same word form appears with different meanings
#     Returns: Dict of {entity_name: list_of_different_entity_urls}
#     """
    
#     # Track entities by their surface form
#     entity_form_to_urls = defaultdict(set)
#     entity_form_to_instances = defaultdict(list)
    
#     for idx, instance in enumerate(test_data):
#         caption = instance[0]
#         image_url = instance[1]
#         entities = instance[3]  # List of entity annotations
        
#         for entity in entities:
#             entity_name = entity[0]  # Surface form (e.g., "Apple")
#             entity_type = entity[1]  # Entity type
#             entity_url = entity[4]   # Wikipedia URL
            
#             # Store this occurrence
#             entity_form_to_urls[entity_name.lower()].add(entity_url)
#             entity_form_to_instances[entity_name.lower()].append({
#                 'instance_idx': idx,
#                 'entity_name': entity_name,
#                 'entity_url': entity_url,
#                 'entity_type': entity_type,
#                 'caption': caption,
#                 'image_url': image_url,
#                 'full_entity_data': entity
#             })
    
#     # Find ambiguous entities (same form, different URLs)
#     ambiguous_entities = {}
#     for entity_form, urls in entity_form_to_urls.items():
#         if len(urls) > 1:
#             ambiguous_entities[entity_form] = {
#                 'urls': list(urls),
#                 'instances': entity_form_to_instances[entity_form]
#             }
    
#     print(f"\n{'='*80}")
#     print("AMBIGUOUS ENTITY ANALYSIS")
#     print('='*80)
#     print(f"Total unique entity forms: {len(entity_form_to_urls)}")
#     print(f"Ambiguous entity forms: {len(ambiguous_entities)}")
#     print(f"Percentage ambiguous: {len(ambiguous_entities)/len(entity_form_to_urls)*100:.1f}%")
    
#     return ambiguous_entities, entity_form_to_instances

# # Find ambiguous pairs
# ambiguous_entities, entity_instances = find_ambiguous_entity_pairs(test_data)

##########################################################
########## Detailed Analysis of Ambiguous Cases###########
##########################################################

import json
from collections import defaultdict
import os

def analyze_wikidiverse_test():
    """Complete analysis in one function"""
    
    # CORRECTED: File is in current directory, not subdirectory
    test_file = "test.json"
    
    # Check if file exists
    if not os.path.exists(test_file):
        print(f"Error: File not found at {test_file}")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        return None, None
    
    # 1. Load data
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"✓ Loaded {len(test_data)} test instances from {test_file}")
    
    # 2. Find ambiguous entities
    entity_form_to_urls = defaultdict(set)
    entity_form_to_instances = defaultdict(list)
    
    total_mentions = 0
    for idx, instance in enumerate(test_data):
        caption = instance[0]
        entities = instance[3]
        total_mentions += len(entities)
        
        for entity in entities:
            entity_name = entity[0].lower()
            entity_url = entity[4]
            
            entity_form_to_urls[entity_name].add(entity_url)
            entity_form_to_instances[entity_name].append({
                'entity_name': entity[0],
                'entity_url': entity_url,
                'entity_type': entity[1],
                'caption': caption,
                'image_url': instance[1],
                'topic': instance[2],
                'start_pos': entity[2],  # Character start position
                'end_pos': entity[3]     # Character end position
            })
    
    # 3. Identify ambiguous cases
    ambiguous = {}
    for entity_name, urls in entity_form_to_urls.items():
        if len(urls) > 1:
            ambiguous[entity_name] = {
                'urls': list(urls),
                'instances': entity_form_to_instances[entity_name],
                'num_instances': len(entity_form_to_instances[entity_name])
            }
    
    print(f"✓ Total entity mentions: {total_mentions}")
    print(f"✓ Unique entity surface forms: {len(entity_form_to_urls)}")
    print(f"✓ Ambiguous entity forms: {len(ambiguous)}")
    print(f"✓ Ambiguity rate: {len(ambiguous)/len(entity_form_to_urls)*100:.1f}%")
    
    # 4. Show statistics
    print(f"\n{'='*60}")
    print("AMBIGUITY STATISTICS")
    print('='*60)
    
    # Count distribution
    ambiguity_counts = defaultdict(int)
    for data in ambiguous.values():
        num_meanings = len(data['urls'])
        ambiguity_counts[num_meanings] += 1
    
    print("\nNumber of different meanings per entity:")
    for num_meanings, count in sorted(ambiguity_counts.items()):
        print(f"  {num_meanings} meanings: {count} entities")
    
    # 5. Find entities with same image but different meanings
    print(f"\n{'='*60}")
    print("TRICKY PAIRS ANALYSIS")
    print('='*60)
    
    # Group instances by image URL
    image_to_entities = defaultdict(lambda: defaultdict(list))
    for entity_name, data in ambiguous.items():
        for inst in data['instances']:
            image_to_entities[inst['image_url']][entity_name].append(inst)
    
    # Find images with multiple ambiguous entities
    tricky_images = []
    for image_url, entities_dict in image_to_entities.items():
        for entity_name, instances in entities_dict.items():
            # Check if this entity has multiple meanings for this image
            unique_urls = set(inst['entity_url'] for inst in instances)
            if len(unique_urls) > 1:
                tricky_images.append({
                    'image_url': image_url,
                    'entity_name': entity_name,
                    'meanings': list(unique_urls),
                    'instances': instances
                })
    
    print(f"Found {len(tricky_images)} 'tricky' cases where same image has captions")
    print(f"with different meanings of the same entity!")
    
    if tricky_images:
        print("\nSample tricky cases:")
        for i, case in enumerate(tricky_images[:5]):
            print(f"\n{i+1}. IMAGE: {case['image_url'].split('/')[-1][:50]}...")
            print(f"   ENTITY: '{case['entity_name'].upper()}' has {len(case['meanings'])} meanings:")
            for j, url in enumerate(case['meanings']):
                # Find example caption for this meaning
                examples = [inst for inst in case['instances'] if inst['entity_url'] == url]
                if examples:
                    caption = examples[0]['caption']
                    # Extract context around the entity
                    start = examples[0]['start_pos']
                    end = examples[0]['end_pos']
                    context = caption[max(0, start-30):min(len(caption), end+30)]
                    print(f"   Meaning {j+1}: {url.split('/')[-1].replace('_', ' ')}")
                    print(f"      Context: ...{context}...")
    
    # 6. Show top ambiguous examples
    print(f"\n{'='*60}")
    print("TOP 20 MOST AMBIGUOUS ENTITIES")
    print('='*60)
    
    sorted_ambig = sorted(ambiguous.items(), 
                         key=lambda x: (len(x[1]['urls']), x[1]['num_instances']), 
                         reverse=True)[:20]
    
    for i, (entity_name, data) in enumerate(sorted_ambig, 1):
        print(f"\n{i}. {entity_name.upper()} ({len(data['urls'])} meanings, {data['num_instances']} mentions):")
        
        # Group by URL to show different meanings
        url_groups = defaultdict(list)
        for inst in data['instances']:
            url_groups[inst['entity_url']].append(inst)
        
        for url_idx, (url, instances) in enumerate(url_groups.items(), 1):
            sample = instances[0]
            entity_title = url.split('/')[-1].replace('_', ' ')
            print(f"   {url_idx}. {entity_title}")
            print(f"      Type: {sample['entity_type']}, Topic: {sample['topic']}")
            
            # Show context around entity
            caption = sample['caption']
            start = sample['start_pos']
            end = sample['end_pos']
            context_start = max(0, start - 30)
            context_end = min(len(caption), end + 30)
            context = caption[context_start:context_end]
            if context_start > 0:
                context = "..." + context
            if context_end < len(caption):
                context = context + "..."
            print(f"      Context: {context}")
    
    return ambiguous, test_data, tricky_images

# Run the analysis
if __name__ == "__main__":
    print("Analyzing WikiDiverse test set for ambiguous entities...")
    print("=" * 60)
    
    ambiguous_results, test_data, tricky_images = analyze_wikidiverse_test()
    
    # Save results
    if ambiguous_results:
        import pickle
        
        # Save full results
        with open('wikidiverse_ambiguous_results.pkl', 'wb') as f:
            pickle.dump({
                'ambiguous_entities': ambiguous_results,
                'tricky_images': tricky_images,
                'test_data_sample': test_data[:5] if test_data else [],
                'statistics': {
                    'total_test_instances': len(test_data),
                    'total_ambiguous': len(ambiguous_results),
                    'tricky_cases': len(tricky_images)
                }
            }, f)
        
        # Save top ambiguous as JSON
        top_ambiguous = {}
        for entity_name, data in list(ambiguous_results.items())[:30]:
            # Get unique meanings with sample contexts
            meanings_info = []
            url_groups = defaultdict(list)
            for inst in data['instances']:
                url_groups[inst['entity_url']].append(inst)
            
            for url, instances in url_groups.items():
                sample = instances[0]
                meanings_info.append({
                    'wikipedia_url': url,
                    'entity_title': url.split('/')[-1].replace('_', ' '),
                    'sample_context': sample['caption'][max(0, sample['start_pos']-50):sample['end_pos']+50],
                    'topic': sample['topic'],
                    'type': sample['entity_type']
                })
            
            top_ambiguous[entity_name] = {
                'num_meanings': len(data['urls']),
                'total_mentions': data['num_instances'],
                'meanings': meanings_info
            }
        
        with open('top_ambiguous_examples.json', 'w') as f:
            json.dump(top_ambiguous, f, indent=2)
        
        # Save tricky cases
        if tricky_images:
            tricky_simple = []
            for case in tricky_images[:10]:  # Save first 10
                tricky_simple.append({
                    'image_filename': case['image_url'].split('/')[-1],
                    'entity': case['entity_name'],
                    'num_meanings': len(case['meanings']),
                    'meanings': [
                        {
                            'url': url,
                            'entity': url.split('/')[-1].replace('_', ' ')
                        }
                        for url in case['meanings']
                    ]
                })
            
            with open('tricky_image_cases.json', 'w') as f:
                json.dump(tricky_simple, f, indent=2)
        
        print(f"\n" + "="*60)
        print("RESULTS SAVED TO:")
        print("="*60)
        print("1. wikidiverse_ambiguous_results.pkl - Full analysis (pickle)")
        print("2. top_ambiguous_examples.json - Top 30 ambiguous entities")
        if tricky_images:
            print("3. tricky_image_cases.json - Cases with same image, different meanings")
        
        print(f"\nSUMMARY:")
        print(f"- Test instances: {len(test_data)}")
        print(f"- Ambiguous entity forms: {len(ambiguous_results)}")
        print(f"- Tricky image cases: {len(tricky_images)}")
        print(f"- These are perfect for testing your network's disambiguation ability!")



# # import json
# # from collections import defaultdict
# # import os

# # def analyze_wikidiverse_test():
# #     """Complete analysis in one function"""
    
# #     # Use the correct path
# #     test_file = "WikiDiverse/test.json"
    
# #     # Check if file exists
# #     if not os.path.exists(test_file):
# #         print(f"Error: File not found at {test_file}")
# #         print("Current directory:", os.getcwd())
# #         print("Files in current directory:", os.listdir('.'))
# #         return None, None
    
# #     # 1. Load data
# #     with open(test_file, 'r') as f:
# #         test_data = json.load(f)
    
# #     print(f"Loaded {len(test_data)} test instances from {test_file}")
    
# #     # 2. Find ambiguous entities
# #     entity_form_to_urls = defaultdict(set)
# #     entity_form_to_instances = defaultdict(list)
    
# #     for idx, instance in enumerate(test_data):
# #         caption = instance[0]
# #         entities = instance[3]
        
# #         for entity in entities:
# #             entity_name = entity[0].lower()
# #             entity_url = entity[4]
            
# #             entity_form_to_urls[entity_name].add(entity_url)
# #             entity_form_to_instances[entity_name].append({
# #                 'entity_name': entity[0],
# #                 'entity_url': entity_url,
# #                 'entity_type': entity[1],
# #                 'caption': caption,
# #                 'image_url': instance[1],
# #                 'topic': instance[2]
# #             })
    
# #     # 3. Identify ambiguous cases
# #     ambiguous = {}
# #     for entity_name, urls in entity_form_to_urls.items():
# #         if len(urls) > 1:
# #             ambiguous[entity_name] = {
# #                 'urls': list(urls),
# #                 'instances': entity_form_to_instances[entity_name],
# #                 'num_instances': len(entity_form_to_instances[entity_name])
# #             }
    
# #     print(f"\nFound {len(ambiguous)} ambiguous entity forms (out of {len(entity_form_to_urls)} total)")
    
# #     # 4. Show statistics
# #     print(f"\n{'='*60}")
# #     print("AMBIGUITY STATISTICS")
# #     print('='*60)
    
# #     # Count distribution
# #     ambiguity_counts = defaultdict(int)
# #     for data in ambiguous.values():
# #         num_meanings = len(data['urls'])
# #         ambiguity_counts[num_meanings] += 1
    
# #     for num_meanings, count in sorted(ambiguity_counts.items()):
# #         print(f"Entities with {num_meanings} meanings: {count}")
    
# #     # 5. Show top examples
# #     print(f"\n{'='*60}")
# #     print("TOP 15 MOST AMBIGUOUS ENTITIES")
# #     print('='*60)
    
# #     sorted_ambig = sorted(ambiguous.items(), 
# #                          key=lambda x: (len(x[1]['urls']), x[1]['num_instances']), 
# #                          reverse=True)[:15]
    
# #     for i, (entity_name, data) in enumerate(sorted_ambig, 1):
# #         print(f"\n{i}. {entity_name.upper()} ({len(data['urls'])} meanings, {data['num_instances']} total mentions):")
        
# #         # Group by URL to show different meanings
# #         url_groups = defaultdict(list)
# #         for inst in data['instances']:
# #             url_groups[inst['entity_url']].append(inst)
        
# #         for url_idx, (url, instances) in enumerate(url_groups.items(), 1):
# #             sample = instances[0]
# #             entity_title = url.split('/')[-1].replace('_', ' ')
# #             print(f"   {url_idx}. {entity_title}")
# #             print(f"      Type: {sample['entity_type']}, Topic: {sample['topic']}")
# #             caption_preview = sample['caption'][:100].replace('\n', ' ')
# #             print(f"      Sample: \"{caption_preview}...\"")
    
# #     return ambiguous, test_data

# # # Run the analysis
# # if __name__ == "__main__":
# #     ambiguous_results, test_data = analyze_wikidiverse_test()
    
# #     # Save results
# #     if ambiguous_results:
# #         import pickle
        
# #         # Save full results
# #         with open('wikidiverse_ambiguous_results.pkl', 'wb') as f:
# #             pickle.dump({
# #                 'ambiguous_entities': ambiguous_results,
# #                 'test_data_sample': test_data[:10] if test_data else []
# #             }, f)
        
# #         # Save as JSON for readability
# #         json_output = {}
# #         for entity_name, data in list(ambiguous_results.items())[:50]:
# #             json_output[entity_name] = {
# #                 'num_meanings': len(data['urls']),
# #                 'meanings': data['urls'],
# #                 'sample_captions': [
# #                     inst['caption'][:150] 
# #                     for inst in data['instances'][:2]
# #                 ]
# #             }
        
# #         with open('top_ambiguous_examples.json', 'w') as f:
# #             json.dump(json_output, f, indent=2)
        
# #         print(f"\nResults saved to:")
# #         print(f"  - wikidiverse_ambiguous_results.pkl (full data)")
# #         print(f"  - top_ambiguous_examples.json (readable examples)")






# import json
# from collections import defaultdict
# import re

# def analyze_ambiguous_cases(ambiguous_entities, entity_instances, top_n=20):
#     """Analyze and display the most interesting ambiguous cases"""
    
#     # Sort by number of different meanings (URLs)
#     sorted_ambiguous = sorted(
#         ambiguous_entities.items(),
#         key=lambda x: len(x[1]['urls']),
#         reverse=True
#     )
    
#     print(f"\n{'='*80}")
#     print(f"TOP {top_n} MOST AMBIGUOUS ENTITIES IN TEST SET")
#     print('='*80)
    
#     results = []
    
#     for i, (entity_form, data) in enumerate(sorted_ambiguous[:top_n]):
#         num_meanings = len(data['urls'])
#         instances = data['instances']
        
#         print(f"\n{i+1}. '{entity_form.upper()}' - {num_meanings} different meanings:")
        
#         # Group instances by entity URL
#         url_to_instances = defaultdict(list)
#         for inst in instances:
#             url_to_instances[inst['entity_url']].append(inst)
        
#         # Show each meaning
#         for url_idx, (url, url_instances) in enumerate(url_to_instances.items()):
#             sample = url_instances[0]
#             print(f"\n   Meaning {url_idx+1}:")
#             print(f"   Wikipedia: {url}")
#             print(f"   Type: {sample['entity_type']}")
#             print(f"   Appears in {len(url_instances)} test instances")
            
#             # Show sample caption
#             caption_preview = sample['caption'][:150].replace('\n', ' ')
#             print(f"   Sample caption: \"{caption_preview}...\"")
            
#             # Extract entity context
#             caption = sample['caption']
#             entity_name = sample['entity_name']
            
#             # Find context around the entity
#             entity_pos = caption.find(entity_name)
#             if entity_pos != -1:
#                 start = max(0, entity_pos - 50)
#                 end = min(len(caption), entity_pos + len(entity_name) + 50)
#                 context = caption[start:end]
#                 print(f"   Context: ...{context}...")
        
#         # Store for later use
#         results.append({
#             'entity_form': entity_form,
#             'num_meanings': num_meanings,
#             'meanings': list(url_to_instances.keys()),
#             'sample_instances': [
#                 {
#                     'caption': inst['caption'][:200],
#                     'entity_url': inst['entity_url'],
#                     'entity_type': inst['entity_type'],
#                     'image_url': inst['image_url']
#                 }
#                 for inst in instances[:3]  # Store first 3 instances
#             ]
#         })
    
#     return results

# # Analyze top ambiguous cases
# top_ambiguous = analyze_ambiguous_cases(ambiguous_entities, entity_instances, top_n=15)


############################################################
######## Tricky" Pairs ####################################
############################################################

# def find_tricky_image_pairs(test_data, ambiguous_entities, entity_instances):
#     """
#     Find cases where the SAME IMAGE has captions with ambiguous entity mentions
#     This is the most challenging case for disambiguation
#     """
    
#     # First, group instances by image URL
#     image_to_instances = defaultdict(list)
#     for idx, instance in enumerate(test_data):
#         image_url = instance[1]
#         image_to_instances[image_url].append({
#             'idx': idx,
#             'caption': instance[0],
#             'entities': instance[3]
#         })
    
#     print(f"\n{'='*80}")
#     print("SEARCHING FOR TRICKY IMAGE PAIRS")
#     print('='*80)
    
#     tricky_pairs = []
    
#     # Look for images with multiple captions containing ambiguous entities
#     for image_url, instances in image_to_instances.items():
#         if len(instances) > 1:
#             # Check if any entity appears ambiguously across these captions
#             all_entities = []
#             for inst in instances:
#                 for entity in inst['entities']:
#                     entity_form = entity[0].lower()
#                     if entity_form in ambiguous_entities:
#                         all_entities.append({
#                             'entity_form': entity_form,
#                             'entity_url': entity[4],
#                             'caption': inst['caption'],
#                             'instance_idx': inst['idx']
#                         })
            
#             # Group by entity form
#             entity_form_groups = defaultdict(list)
#             for ent in all_entities:
#                 entity_form_groups[ent['entity_form']].append(ent)
            
#             # Find cases where same entity form has different URLs
#             for entity_form, entries in entity_form_groups.items():
#                 unique_urls = set(entry['entity_url'] for entry in entries)
#                 if len(unique_urls) > 1:
#                     # This is a tricky pair!
#                     tricky_pairs.append({
#                         'image_url': image_url,
#                         'entity_form': entity_form,
#                         'meanings': list(unique_urls),
#                         'instances': entries
#                     })
    
#     print(f"\nFound {len(tricky_pairs)} 'tricky' image pairs!")
    
#     # Display the most interesting ones
#     for i, pair in enumerate(tricky_pairs[:10]):
#         print(f"\n{i+1}. IMAGE: {pair['image_url']}")
#         print(f"   Entity: '{pair['entity_form'].upper()}' has {len(pair['meanings'])} meanings:")
        
#         for url in pair['meanings']:
#             # Find a caption example for this URL
#             examples = [e for e in pair['instances'] if e['entity_url'] == url]
#             if examples:
#                 example = examples[0]
#                 caption_preview = example['caption'][:120].replace('\n', ' ')
#                 print(f"\n   Meaning: {url}")
#                 print(f"   Caption: \"{caption_preview}...\"")
    
#     return tricky_pairs

# # Find tricky image pairs
# tricky_pairs = find_tricky_image_pairs(test_data, ambiguous_entities, entity_instances)




##########################################################
############### Top 50 ambiguous entities ################
##########################################################

# from collections import defaultdict
# import json

# def parse_entity_images_enhanced(file_path, max_lines=None):
#     """Enhanced parser with progress tracking"""
#     entity_to_images = defaultdict(list)
#     line_count = 0
    
#     with open(file_path, 'r', encoding='utf-8') as f:
#         header = f.readline().strip()
#         print(f"Header: {header}")
        
#         for line_num, line in enumerate(f, start=2):
#             if max_lines and line_num > max_lines:
#                 break
                
#             line = line.strip()
#             if not line:
#                 continue
            
#             # Progress indicator
#             if line_num % 10000 == 0:
#                 print(f"Processing line {line_num}...")
            
#             # Split by @@@@
#             if '@@@@' in line:
#                 parts = line.split('@@@@', 1)
#                 if len(parts) == 2:
#                     entity = parts[0].strip()
#                     images_str = parts[1].strip()
                    
#                     # Parse images
#                     images = []
#                     if images_str:
#                         images = [img.strip() for img in images_str.split('[AND]') if img.strip()]
                    
#                     entity_to_images[entity] = images
#                     line_count += 1
    
#     print(f"\nParsed {line_count} entities from {line_num-1} lines")
#     return entity_to_images

# def analyze_ambiguity(entity_images, search_terms=None):
#     """Comprehensive ambiguity analysis"""
    
#     # Group by clean names
#     name_to_variants = defaultdict(list)
#     for entity, images in entity_images.items():
#         clean_name = entity.split('(')[0].strip().lower()
#         name_to_variants[clean_name].append({
#             'full_name': entity,
#             'image_count': len(images),
#             'images': images[:2] if images else []  # Store first 2 images for reference
#         })
    
#     # Find ambiguous names
#     ambiguous = {name: variants for name, variants in name_to_variants.items() 
#                  if len(variants) > 1}
    
#     print(f"\n{'='*80}")
#     print("AMBIGUITY ANALYSIS REPORT")
#     print('='*80)
#     print(f"Total unique base names: {len(name_to_variants)}")
#     print(f"Ambiguous base names: {len(ambiguous)} ({len(ambiguous)/len(name_to_variants)*100:.1f}%)")
    
#     # Show statistics
#     ambiguity_counts = defaultdict(int)
#     for name, variants in ambiguous.items():
#         ambiguity_counts[len(variants)] += 1
    
#     print(f"\nAmbiguity distribution:")
#     for count, freq in sorted(ambiguity_counts.items()):
#         print(f"  {count} variants: {freq} names")
    
#     # Search for specific terms if provided
#     if search_terms:
#         print(f"\n{'='*80}")
#         print("SEARCH RESULTS FOR SPECIFIC TERMS")
#         print('='*80)
        
#         for term in search_terms:
#             term_lower = term.lower()
#             if term_lower in name_to_variants:
#                 variants = name_to_variants[term_lower]
#                 print(f"\n{term.upper()} ({len(variants)} variants):")
#                 for i, var in enumerate(variants, 1):
#                     print(f"  {i}. {var['full_name']}")
#                     print(f"     Images: {var['image_count']}")
#                     if var['images']:
#                         # Extract meaningful info from image URLs
#                         first_img = var['images'][0]
#                         filename = first_img.split('/')[-1].split('.')[0].replace('_', ' ')
#                         print(f"     Hint: {filename[:50]}...")
#             else:
#                 print(f"\n{term.upper()}: No ambiguous variants found")
    
#     # Show top ambiguous examples
#     print(f"\n{'='*80}")
#     print("TOP 15 MOST AMBIGUOUS ENTITIES")
#     print('='*80)
    
#     # Sort by number of variants
#     sorted_ambiguous = sorted(ambiguous.items(), key=lambda x: len(x[1]), reverse=True)
    
#     for i, (name, variants) in enumerate(sorted_ambiguous[:15], 1):
#         print(f"\n{i}. {name.upper()} ({len(variants)} Wikipedia pages):")
#         for j, var in enumerate(variants[:3], 1):  # Show first 3 variants
#             print(f"   {j}. {var['full_name']} ({var['image_count']} images)")
    
#     return ambiguous, name_to_variants

# # MAIN EXECUTION
# if __name__ == "__main__":
#     # Parse the data
#     print("Parsing wikipedia_entity2imgs.tsv...")
#     entity_images = parse_entity_images_enhanced("wikipedia_entity2imgs.tsv", max_lines=None)
    
#     # Analyze ambiguity
#     search_terms = ['apple', 'jaguar', 'python', 'mercury', 'springfield', 'eagle', 'taylor']
#     ambiguous, name_to_variants = analyze_ambiguity(entity_images, search_terms)
    
#     # Save results
#     import pickle
#     with open('entity_analysis_results.pkl', 'wb') as f:
#         pickle.dump({
#             'entity_images': dict(entity_images),
#             'ambiguous': ambiguous,
#             'name_to_variants': dict(name_to_variants)
#         }, f)
    
#     print(f"\nAnalysis complete! Results saved to 'entity_analysis_results.pkl'")
    
#     # Generate a JSON report of top ambiguous entities
#     top_ambiguous = {}
#     for name, variants in list(ambiguous.items())[:50]:
#         top_ambiguous[name] = [v['full_name'] for v in variants]
    
#     with open('top_ambiguous_entities.json', 'w') as f:
#         json.dump(top_ambiguous, f, indent=2)
    
#     print("Top 50 ambiguous entities saved to 'top_ambiguous_entities.json'")