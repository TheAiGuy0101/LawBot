import PyPDF2
import re
import os
import io

def split_pdf_content(input_file, output_folder, split_phrase):
    try:
        # Open the PDF file
        with open(input_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Create the output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Initialize variables
            current_writer = PyPDF2.PdfWriter()
            file_count = 0
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Find all occurrences of the split phrase on this page
                split_positions = [m.start() for m in re.finditer(re.escape(split_phrase), text)]
                
                if not split_positions:
                    # If no split phrase on this page, add the whole page
                    current_writer.add_page(page)
                else:
                    # Process each split on the page
                    last_pos = 0
                    for pos in split_positions:
                        # Create a new page with content up to the split phrase
                        temp_writer = PyPDF2.PdfWriter()
                        temp_writer.add_page(page)
                        temp_stream = io.BytesIO()
                        temp_writer.write(temp_stream)
                        temp_stream.seek(0)
                        
                        temp_reader = PyPDF2.PdfReader(temp_stream)
                        new_page = temp_reader.pages[0]
                        
                        # Add the partial page to the current writer
                        current_writer.add_page(new_page)
                        
                        # Save the current section
                        file_count += 1
                        output_file = os.path.join(output_folder, f"section_{file_count}.pdf")
                        with open(output_file, 'wb') as out_file:
                            current_writer.write(out_file)
                        print(f"Created file: {output_file}")
                        
                        # Start a new section
                        current_writer = PyPDF2.PdfWriter()
                        
                        last_pos = pos + len(split_phrase)
                    
                    # If there's remaining content on the page, add it to the current writer
                    if last_pos < len(text):
                        temp_writer = PyPDF2.PdfWriter()
                        temp_writer.add_page(page)
                        temp_stream = io.BytesIO()
                        temp_writer.write(temp_stream)
                        temp_stream.seek(0)
                        
                        temp_reader = PyPDF2.PdfReader(temp_stream)
                        new_page = temp_reader.pages[0]
                        current_writer.add_page(new_page)
            
            # Save the last section if it's not empty
            if len(current_writer.pages) > 0:
                file_count += 1
                output_file = os.path.join(output_folder, f"section_{file_count}.pdf")
                with open(output_file, 'wb') as out_file:
                    current_writer.write(out_file)
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