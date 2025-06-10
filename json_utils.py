import json
from typing import List, Dict, Any
import os

def remove_duplicate_courses(input_file: str, output_file: str = None) -> List[Dict[str, Any]]:
    """
    Read a JSON file containing course records and remove duplicates by course_id.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Optional path to save the deduplicated JSON. If not provided,
                   will save to the same directory as input with '_deduped' suffix.
                   
    Returns:
        List of deduplicated course records
    """
    # Set default output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_deduped{ext}"
    
    # Read the input JSON file
    print(f"Reading courses from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    
    # Track seen course_ids and deduplicate
    seen = set()
    deduped_courses = []
    duplicate_count = 0
    
    for course in courses:
        course_id = course.get('course_id')
        if not course_id:
            print(f"Warning: Found course without course_id: {course}")
            continue
            
        if course_id not in seen:
            seen.add(course_id)
            deduped_courses.append(course)
        else:
            duplicate_count += 1
    
    # Save the deduplicated courses
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduped_courses, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(courses)} courses, found {duplicate_count} duplicates")
    print(f"Saved {len(deduped_courses)} unique courses to {output_file}")
    
    return deduped_courses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove duplicate courses from a JSON file')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Path to output JSON file (optional)')
    
    args = parser.parse_args()
    remove_duplicate_courses(args.input_file, args.output)
