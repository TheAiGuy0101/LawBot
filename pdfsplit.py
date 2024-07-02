import PyPDF2
import re
import os

def split_pdf_content(input_file, output_folder, split_phrase):
    try:
        # Open the PDF file
        with open(input_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
            
            # Split the content based on the phrase
            sections = re.split(f'({re.escape(split_phrase)})', content)
            
            # Create the output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Write each section to a separate file
            file_count = 0
            current_section = ""
            for section in sections:
                if section.strip() == split_phrase:
                    if current_section:
                        file_count += 1
                        output_file = os.path.join(output_folder, f"section_{file_count}.txt")
                        with open(output_file, 'w', encoding='utf-8') as out_file:
                            out_file.write(current_section.strip())
                        print(f"Created file: {output_file}")
                    current_section = split_phrase
                else:
                    current_section += section
            
            # Write the last section if it exists
            if current_section:
                file_count += 1
                output_file = os.path.join(output_folder, f"section_{file_count}.txt")
                with open(output_file, 'w', encoding='utf-8') as out_file:
                    out_file.write(current_section.strip())
                print(f"Created file: {output_file}")
            
            print(f"Total sections created: {file_count}")
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except PyPDF2.errors.PdfReadError:
        print(f"Error: Unable to read the PDF file '{input_file}'. It may be corrupted or password-protected.")
    except PermissionError:
        print(f"Error: Permission denied when trying to create or write to the output folder '{output_folder}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# Usage
input_file = "Aug2022.pdf"
output_folder = "split_sections"
split_phrase = "(2022) 8 ILRA"

split_pdf_content(input_file, output_folder, split_phrase)