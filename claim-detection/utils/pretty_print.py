import pandas as pd
import colorama
from colorama import Fore, Style
from tabulate import tabulate
from pathlib import Path
from io import StringIO

# Initialize colorama for colored terminal output
colorama.init()

# Set pandas display options
pd.set_option('display.max_colwidth', 40)  # Limit column width for better display
pd.set_option('display.unicode.east_asian_width', True)  # Better handling of Thai characters


def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n{Fore.CYAN}▶ {title}:{Style.RESET_ALL}")


def print_table(df, headers="keys", tablefmt="grid", **kwargs):
    """Print a well-formatted table that works with console width"""
    try:
        print(tabulate(df, headers=headers, tablefmt=tablefmt, **kwargs))
    except Exception as e:
        print(f"  {Fore.RED}Error displaying table: {e}{Style.RESET_ALL}")
        try:
            # Fallback to a simpler format
            print(df)
        except:
            print(f"  {Fore.RED}Unable to display data in table format{Style.RESET_ALL}")


def print_report_header(title="DATASET ANALYSIS REPORT"):
    """Print a formatted report header"""
    print(f"\n{Fore.GREEN}═══════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.GREEN}          {title}                {Style.RESET_ALL}")
    print(f"{Fore.GREEN}═══════════════════════════════════════════════════{Style.RESET_ALL}")


def print_report_footer(title="ANALYSIS COMPLETE"):
    """Print a formatted report footer"""
    print(f"\n{Fore.GREEN}═══════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.GREEN}          {title}                      {Style.RESET_ALL}")
    print(f"{Fore.GREEN}═══════════════════════════════════════════════════{Style.RESET_ALL}")


def print_label_distribution(df, split_name=None):
    """
    Print the label distribution for a dataframe with percentage information
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing image data with a 'label' column
    split_name : str, optional
        The name of the data split (e.g., 'Train', 'Validation', 'Test')
        If provided, it will be included in the header
    """
    if split_name:
        print_section_header(f"{split_name} set label distribution")
    else:
        print_section_header("Label distribution")
        
    if 'label' not in df.columns:
        print(f"  {Fore.YELLOW}No 'label' column found in the dataset{Style.RESET_ALL}")
        return
        
    label_counts = df["label"].value_counts()
    label_df = pd.DataFrame({
        'Label': label_counts.index,
        'Count': label_counts.values,
        'Percentage': [(count/len(df)*100) for count in label_counts.values]
    })
    label_df['Percentage'] = label_df['Percentage'].apply(lambda x: f"{x:.1f}%")
    
    print_table(label_df, showindex=False)


def analyze_image_dataset(df, script_dir, img_dir):
    """
    Print a comprehensive analysis of an image dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing image data with at least 'path' and 'label' columns
    script_dir : pathlib.Path
        The current working directory
    img_dir : pathlib.Path
        The directory containing the images
    """
    # Print headers for report
    print_report_header()

    print_section_header("Current Working Directory")
    print(f"  {script_dir}")

    # Add summary of found images
    if not df.empty:
        claimed_count = df[df['label'] == 'claimed'].shape[0]
        unclaimed_count = df[df['label'] == 'unclaimed'].shape[0]
        
        print_section_header("Image Summary")
        print(f"  • Total images: {Fore.YELLOW}{len(df)}{Style.RESET_ALL}")
        print(f"  • Claimed images: {Fore.YELLOW}{claimed_count}{Style.RESET_ALL} ({claimed_count/len(df)*100:.1f}%)")
        print(f"  • Unclaimed images: {Fore.YELLOW}{unclaimed_count}{Style.RESET_ALL} ({unclaimed_count/len(df)*100:.1f}%)")

        # Print label distribution
        print_section_header("Label Distribution")
        if 'label' in df.columns:
            label_counts = df.groupby('label').size().reset_index(name='Count')
            label_counts['Percentage'] = (label_counts['Count'] / label_counts['Count'].sum() * 100).round(1).astype(str) + '%'
            print_table(label_counts, showindex=False)
        else:
            print(f"  {Fore.YELLOW}No 'label' column found in the dataset{Style.RESET_ALL}")
        
        # Print sample data
        print_section_header("Sample Data (First 5 Rows)")
        sample_df = df[['filename', 'label']].head() if 'filename' in df.columns else df.head()
        print_table(sample_df, showindex=True)

        # Print dataframe shape
        print_section_header("Dataset Shape")
        print(f"  Rows: {Fore.YELLOW}{df.shape[0]}{Style.RESET_ALL}, Columns: {Fore.YELLOW}{df.shape[1]}{Style.RESET_ALL}")

        # Print columns list
        print_section_header("Dataset Columns")
        for col in df.columns:
            print(f"  • {col}")

        # Print dataframe info in a more readable format
        print_section_header("Dataset Info")
        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        print('  ' + info_str.replace('\n', '\n  '))

        # Print missing values
        print_section_header("Missing Values")
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            null_df = pd.DataFrame({
                'Column': null_counts.index,
                'Missing Values': null_counts.values,
                'Percentage': (null_counts / len(df) * 100).round(2).astype(str) + '%'
            })
            print_table(null_df, showindex=False)
        else:
            print(f"  {Fore.GREEN}No missing values found!{Style.RESET_ALL}")

        # Print duplicate count
        print_section_header("Duplicated Rows")
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            print(f"  {Fore.YELLOW}Found {dup_count} duplicated rows ({dup_count/len(df)*100:.2f}% of dataset){Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}No duplicated rows found!{Style.RESET_ALL}")

        # Show top image file patterns
        print_section_header("Top Image Filename Patterns")
        if 'filename' in df.columns:
            # Extract filename pattern (before extension)
            df_copy = df.copy()
            df_copy['filename_pattern'] = df_copy['filename'].apply(lambda x: x.split('.')[0].rstrip('0123456789') if '.' in x else x)
            pattern_counts = df_copy['filename_pattern'].value_counts().reset_index()
            pattern_counts.columns = ['Pattern', 'Count']
            pattern_counts['Percentage'] = (pattern_counts['Count'] / len(df_copy) * 100).round(1).astype(str) + '%'
            
            if not pattern_counts.empty:
                print_table(pattern_counts.head(10), showindex=False)
            else:
                print(f"  {Fore.YELLOW}No filename patterns to display{Style.RESET_ALL}")
        
        # Top folder analysis
        print_section_header("Folder Distribution")
        if 'path' in df.columns:
            # Extract parent folder name from path
            df_copy = df.copy()
            df_copy['parent_folder'] = df_copy['path'].apply(lambda x: Path(x).parent.name)
            folder_counts = df_copy['parent_folder'].value_counts().reset_index()
            folder_counts.columns = ['Folder', 'Count']
            folder_counts['Percentage'] = (folder_counts['Count'] / len(df_copy) * 100).round(1).astype(str) + '%'
            
            if not folder_counts.empty:
                print_table(folder_counts.head(10), showindex=False)
            else:
                print(f"  {Fore.YELLOW}No folder distribution to display{Style.RESET_ALL}")
    else:
        # Show directory structure if no data was found
        print_section_header("Directory Structure")
        if img_dir.exists():
            print(f"  {Fore.YELLOW}Directory exists but no images were found{Style.RESET_ALL}")
            print(f"\n  Directory contents of {img_dir}:")
            for item in img_dir.iterdir():
                item_type = f"{Fore.BLUE}directory{Style.RESET_ALL}" if item.is_dir() else f"{Fore.CYAN}file{Style.RESET_ALL}"
                print(f"  • {item.name} ({item_type})")
                
                # If it's a directory, show first level of contents
                if item.is_dir():
                    try:
                        subcontents = list(item.iterdir())
                        if subcontents:
                            for subitem in subcontents[:5]:  # Show at most 5 subitems
                                subtype = f"{Fore.BLUE}directory{Style.RESET_ALL}" if subitem.is_dir() else f"{Fore.CYAN}file{Style.RESET_ALL}"
                                print(f"    └─ {subitem.name} ({subtype})")
                            if len(subcontents) > 5:
                                print(f"    └─ ... and {len(subcontents) - 5} more items")
                        else:
                            print(f"    └─ {Fore.YELLOW}Empty directory{Style.RESET_ALL}")
                    except PermissionError:
                        print(f"    └─ {Fore.RED}Permission denied{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}Directory does not exist: {img_dir}{Style.RESET_ALL}")
            
            # Show parent directory to help troubleshoot
            if img_dir.parent.exists():
                print(f"\n  Parent directory contents ({img_dir.parent}):")
                for item in img_dir.parent.iterdir():
                    item_type = f"{Fore.BLUE}directory{Style.RESET_ALL}" if item.is_dir() else f"{Fore.CYAN}file{Style.RESET_ALL}"
                    print(f"  • {item.name} ({item_type})")

    # Final summary
    print_report_footer() 