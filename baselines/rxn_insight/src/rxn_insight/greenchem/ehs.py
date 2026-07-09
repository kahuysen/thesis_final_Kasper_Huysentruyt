import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Union
import os, json, logging, time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_ghs_information(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract GHS classification data from ECHA source."""
    ghs_data = {
        'cid': None,
        'ec_number': None,
        'pictograms': [],
        'signal': None,
        'hazard_statements': [],
        'precautionary_codes': [],
        'echa_summary': None,
        'raw_data': None
    }

    try:
        # Get CID
        ghs_data['cid'] = json_data.get('Record', {}).get('RecordNumber')

        # First, find ECHA reference number
        echa_ref_num = None
        references = json_data.get('Record', {}).get('Reference', [])
        for ref in references:
            if ref.get('SourceName') == 'European Chemicals Agency (ECHA)':
                echa_ref_num = ref.get('ReferenceNumber')
                ghs_data['ec_number'] = ref.get('SourceID')
                break

        if echa_ref_num is None:
            ghs_data['error'] = 'No ECHA data found'
            return ghs_data

        # Navigate to GHS Classification section
        sections = json_data.get('Record', {}).get('Section', [])

        for main_section in sections:
            if main_section.get('TOCHeading') == 'Safety and Hazards':
                for sub_section in main_section.get('Section', []):
                    if sub_section.get('TOCHeading') == 'Hazards Identification':
                        for ghs_section in sub_section.get('Section', []):
                            if ghs_section.get('TOCHeading') == 'GHS Classification':
                                # Extract information
                                for info in ghs_section.get('Information', []):
                                    # Check if this is from ECHA (using dynamic reference number)
                                    if info.get('ReferenceNumber') == echa_ref_num:
                                        name = info.get('Name', '')

                                        if name == 'Pictogram(s)':
                                            pictograms = extract_pictograms(info)
                                            ghs_data['pictograms'] = pictograms

                                        elif name == 'Signal':
                                            signal = extract_text_value(info)
                                            ghs_data['signal'] = signal

                                        elif name == 'GHS Hazard Statements':
                                            statements = extract_hazard_statements(info)
                                            ghs_data['hazard_statements'] = statements

                                        elif name == 'Precautionary Statement Codes':
                                            codes = extract_text_value(info)
                                            ghs_data['precautionary_codes'] = codes

                                        elif name == 'ECHA C&L Notifications Summary':
                                            summary = extract_text_value(info)
                                            ghs_data['echa_summary'] = summary

    except Exception as e:
        print(f"Error extracting GHS data: {e}")
        ghs_data['error'] = str(e)

    return ghs_data


def extract_pictograms(info: Dict[str, Any]) -> List[str]:
    """Extract pictogram information."""
    pictograms = []
    value = info.get('Value', {})
    string_markups = value.get('StringWithMarkup', [])

    for markup_item in string_markups:
        markups = markup_item.get('Markup', [])
        for markup in markups:
            if markup.get('Type') == 'Icon':
                pictograms.append(markup.get('Extra', ''))

    return pictograms


def extract_hazard_statements(info: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract hazard statements with codes and descriptions."""
    statements = []
    value = info.get('Value', {})
    string_markups = value.get('StringWithMarkup', [])

    for markup_item in string_markups:
        statement_text = markup_item.get('String', '')
        # Parse format: "H315 (100%): Causes skin irritation [Warning Skin corrosion/irritation]"
        if statement_text and ':' in statement_text:
            parts = statement_text.split(':', 1)
            code_part = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ''

            # Extract code and percentage
            code = code_part.split()[0] if code_part else ''
            percentage = code_part[len(code):].strip() if code else ''

            statements.append({
                'code': code,
                'percentage': percentage,
                'description': description
            })

    return statements


def extract_text_value(info: Dict[str, Any]) -> str:
    """Extract simple text value from info."""
    value = info.get('Value', {})
    string_markups = value.get('StringWithMarkup', [])

    if string_markups:
        return string_markups[0].get('String', '')
    return ''


def fetch_single_cid(cid: int, session: Optional[requests.Session]) -> Dict[str, Any]:
    """Fetch GHS data for a single CID."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON/?response_type=display&heading=GHS%20Classification"
    url = base_url.format(cid)
    if session is None:
        session = create_session()

    try:
        response = session.get(url, timeout=30)

        if response.status_code == 200:
            json_data = response.json()
            ghs_data = extract_ghs_information(json_data)
            ghs_data['status'] = 'success'
            return ghs_data
        elif response.status_code == 404:
            return {
                'cid': cid,
                'status': 'not_found',
                'error': 'No GHS classification data found'
            }
        else:
            return {
                'cid': cid,
                'status': 'error',
                'error': f'HTTP {response.status_code}'
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for CID {cid}: {e}")
        return {
            'cid': cid,
            'status': 'error',
            'error': str(e)
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for CID {cid}: {e}")
        return {
            'cid': cid,
            'status': 'error',
            'error': 'Invalid JSON response'
        }


def create_session() -> requests.Session:
    """Create a session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class PubChemGHSScraper:
    """Efficient scraper for PubChem GHS classification data from ECHA."""

    def __init__(self, max_workers: int = 5, delay_between_requests: float = 0.2):
        """
        Initialize the scraper.

        Args:
            max_workers: Maximum number of concurrent requests
            delay_between_requests: Delay between requests in seconds
        """
        self.max_workers = max_workers
        self.delay = delay_between_requests
        self.session = create_session()
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON/?response_type=display&heading=GHS%20Classification"

    def fetch_single_cid(self, cid: int) -> Dict[str, Any]:
        """Fetch GHS data for a single CID."""
        return fetch_single_cid(cid, self.session)

    def fetch_multiple_cids(self, cids: List[int], save_progress: bool = True,
                            output_file: str = 'ghs_data.csv') -> pd.DataFrame:
        """
        Fetch GHS data for multiple CIDs with progress saving.

        Args:
            cids: List of PubChem CID numbers
            save_progress: Whether to save progress periodically
            output_file: Output filename for saving progress

        Returns:
            DataFrame with GHS classification data
        """
        total = len(cids)

        logger.info(f"Starting to fetch GHS data for {total} CIDs")

        # Load existing progress if file exists
        existing_results = []
        existing_cids = set()
        if save_progress and os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                existing_cids = set(existing_df['cid'].tolist())
                existing_results = existing_df.to_dict('records')
                logger.info(f"Loaded {len(existing_cids)} existing records")
            except Exception as e:
                logger.warning(f"Could not load existing file: {e}")

        # Filter out already processed CIDs
        remaining_cids = [cid for cid in cids if cid not in existing_cids]

        if not remaining_cids:
            logger.info("All CIDs already processed")
            return pd.DataFrame(existing_results)

        logger.info(f"Processing {len(remaining_cids)} remaining CIDs")

        # Process only new CIDs
        new_results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_cid = {
                executor.submit(self._fetch_with_delay, cid): cid
                for cid in remaining_cids
            }

            # Process completed tasks
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                try:
                    result = future.result()
                    new_results.append(result)
                    completed += 1

                    # Log progress
                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{len(remaining_cids)} new CIDs processed")

                    # Save progress periodically (append new results to existing)
                    if save_progress and completed % 50 == 0:
                        all_results = existing_results + new_results
                        self._save_results(all_results, output_file)

                except Exception as e:
                    logger.error(f"Error processing CID {cid}: {e}")
                    new_results.append({
                        'cid': cid,
                        'status': 'error',
                        'error': str(e)
                    })

        # Final save with all results
        all_results = existing_results + new_results
        if save_progress:
            self._save_results(all_results, output_file)

        logger.info(f"Completed fetching data. Total processed: {len(all_results)}")
        return pd.DataFrame(all_results)

    def _fetch_with_delay(self, cid: int) -> Dict[str, Any]:
        """Fetch with delay to respect rate limits."""
        time.sleep(self.delay)
        return self.fetch_single_cid(cid)

    def _save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save results to CSV file."""
        df = pd.DataFrame(results)

        # Flatten hazard statements for CSV
        if 'hazard_statements' in df.columns:
            df['hazard_codes'] = df['hazard_statements'].apply(
                lambda x: '|'.join([s['code'] for s in x]) if isinstance(x, list) else ''
            )
            df['hazard_descriptions'] = df['hazard_statements'].apply(
                lambda x: '|'.join([s['description'] for s in x]) if isinstance(x, list) else ''
            )

        # Convert lists to strings for CSV
        if 'pictograms' in df.columns:
            df['pictograms'] = df['pictograms'].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else ''
            )

        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} records to {filename}")


class EHSScoring:
    """
    Environmental, Health, and Safety scoring system for chemicals based on GHS codes
    and physical properties.
    """

    def __init__(self):
        # Define hazard categories for health
        self.black_health_labels = ["H300", "H310", "H330", "H340", "H350", "H360"]
        self.red_health_labels = ["H314"]
        self.orange_health_labels = ["H301", "H311", "H331", "H341", "H351", "H361", "H370", "H372"]
        self.yellow_health_labels = ["H318", "H334"]
        self.green_health_labels = ["H302", "H304", "H315", "H317", "H319", "H332", "H335", "H336", "H371", "H373"]

        # Define hazard categories for environment
        self.black_env_labels = ["H420"]
        self.red_env_labels = ["H400", "H410", "H411"]
        self.yellow_env_labels = ["H412", "H413"]

        # All H-codes for reference
        self.all_health_labels = (self.black_health_labels + self.red_health_labels +
                                  self.orange_health_labels + self.yellow_health_labels +
                                  self.green_health_labels)
        self.all_env_labels = self.black_env_labels + self.red_env_labels + self.yellow_env_labels

    def calculate_health_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate health scores based on H-codes present in the DataFrame.

        Args:
            df: DataFrame with H-code columns

        Returns:
            DataFrame with health score columns added
        """
        df = df.copy()

        # Count labels in each category
        df['black_health_count'] = df[self.black_health_labels].sum(axis=1)
        df['red_health_count'] = df[self.red_health_labels].sum(axis=1)
        df['orange_health_count'] = df[self.orange_health_labels].sum(axis=1)
        df['yellow_health_count'] = df[self.yellow_health_labels].sum(axis=1)
        df['green_health_count'] = df[self.green_health_labels].sum(axis=1)

        # Calculate health score (higher score = more hazardous)
        df['health_score'] = 1  # Default to lowest hazard

        df.loc[df['green_health_count'] > 0, 'health_score'] = 2
        df.loc[df['yellow_health_count'] > 0, 'health_score'] = 4
        df.loc[df['orange_health_count'] > 0, 'health_score'] = 6
        df.loc[df['red_health_count'] > 0, 'health_score'] = 7
        df.loc[df['black_health_count'] > 0, 'health_score'] = 9

        return df

    def calculate_environment_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate environmental scores based on H-codes.

        Args:
            df: DataFrame with H-code columns

        Returns:
            DataFrame with environment score columns added
        """
        df = df.copy()

        # Count labels in each category
        df['black_env_count'] = df[self.black_env_labels].sum(axis=1)
        df['red_env_count'] = df[self.red_env_labels].sum(axis=1)
        df['yellow_env_count'] = df[self.yellow_env_labels].sum(axis=1)

        # Calculate environment score (higher score = more hazardous)
        df['env_score'] = 3  # Default to low hazard

        df.loc[df['yellow_env_count'] > 0, 'env_score'] = 5
        df.loc[df['red_env_count'] > 0, 'env_score'] = 7
        df.loc[df['black_env_count'] > 0, 'env_score'] = 10

        return df

    def calculate_safety_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate safety scores based on physical properties.

        Args:
            df: DataFrame with physical property columns

        Returns:
            DataFrame with safety score columns added
        """
        df = df.copy()

        # Flash point score (FP assumed to be in Celsius)
        df['fp_score'] = 0  # Default
        if 'FP' in df.columns:
            # Convert thresholds from Kelvin to Celsius
            df.loc[df['FP'] < -20, 'fp_score'] = 7  # < 253.15K
            df.loc[(df['FP'] >= -20) & (df['FP'] < 0), 'fp_score'] = 5  # 253.15-273.15K
            df.loc[(df['FP'] >= 0) & (df['FP'] < 24), 'fp_score'] = 4  # 273.15-297.15K
            df.loc[(df['FP'] >= 24) & (df['FP'] < 60), 'fp_score'] = 3  # 297.15-333.15K
            df.loc[df['FP'] >= 60, 'fp_score'] = 1  # >= 333.15K

        # Auto-ignition temperature score (AIT assumed to be in Celsius)
        df['ait_score'] = 0  # Default
        if 'AIT' in df.columns:
            df.loc[df['AIT'] <= 200, 'ait_score'] = 1  # <= 473.15K

        # Boiling point volatility score (ExpBP assumed to be in Celsius)
        df['bp_score'] = 0  # Default
        if 'ExpBP' in df.columns:
            df.loc[df['ExpBP'] < 50, 'bp_score'] = 7  # < 323.15K
            df.loc[(df['ExpBP'] >= 50) & (df['ExpBP'] < 70), 'bp_score'] = 5  # 323.15-343.15K
            df.loc[(df['ExpBP'] >= 70) & (df['ExpBP'] < 140), 'bp_score'] = 3  # 343.15-413.15K
            df.loc[(df['ExpBP'] >= 140) & (df['ExpBP'] < 200), 'bp_score'] = 5  # 413.15-473.15K
            df.loc[df['ExpBP'] >= 200, 'bp_score'] = 7  # >= 473.15K

        # Peroxide formation score (binary: 0 or 1)
        df['peroxide_score'] = 0  # Default
        if 'peroxide' in df.columns:
            df['peroxide_score'] = df['peroxide'].astype(int)

        # Resistivity score (binary: 0 or 1)
        df['resistivity_score'] = 0  # Default
        if 'resistivity' in df.columns:
            df['resistivity_score'] = df['resistivity'].astype(int)

        # Combined safety score (sum of all scores)
        df['safety_score'] = (df['fp_score'] + df['ait_score'] +
                              df['peroxide_score'] + df['resistivity_score'])

        return df

    def calculate_ehs_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall EHS ranking based on all scores.

        Args:
            df: DataFrame with all score columns

        Returns:
            DataFrame with EHS ranking added
        """
        df = df.copy()

        # Correct health score based on boiling point
        # Add 1 to health score if BP >= 85°C (358.15K)
        df['corrected_health_score'] = df['health_score']
        if 'ExpBP' in df.columns:
            df.loc[df['ExpBP'] >= 85, 'corrected_health_score'] = df['health_score'] + 1

        # Correct environment score (take max of env_score and bp_score)
        df['corrected_env_score'] = df[['env_score', 'bp_score']].max(axis=1)

        # Determine overall ranking
        df['ehs_ranking'] = 'recommended'  # Default

        # Check each row for hazard levels
        for idx in df.index:
            scores = np.array([
                df.loc[idx, 'safety_score'],
                df.loc[idx, 'corrected_health_score'],
                df.loc[idx, 'corrected_env_score']
            ])

            # Hazardous if any score >= 8, or if more than one score >= 7
            if (scores >= 8).any() or ((scores >= 7).sum() > 1):
                df.loc[idx, 'ehs_ranking'] = 'hazardous'
            # Problematic if max score is 7, or if more than one score is 4-6
            elif (scores.max() == 7) or (((scores >= 4) & (scores <= 6)).sum() > 1):
                df.loc[idx, 'ehs_ranking'] = 'problematic'
            else:
                df.loc[idx, 'ehs_ranking'] = 'recommended'

        return df

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to calculate all scores for a DataFrame.

        Args:
            df: Input DataFrame with H-codes and physical properties

        Returns:
            DataFrame with all scores and rankings added
        """
        # Check for required columns
        h_codes = [col for col in df.columns if col.startswith('H') and col[1:].isdigit()]
        if not h_codes:
            raise ValueError("No H-code columns found in DataFrame")

        # Calculate all scores
        df = self.calculate_health_score(df)
        df = self.calculate_environment_score(df)
        df = self.calculate_safety_score(df)
        df = self.calculate_ehs_ranking(df)

        return df

    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """
        Get summary statistics of EHS rankings.

        Args:
            df: DataFrame with EHS rankings

        Returns:
            Dictionary with summary statistics
        """
        if 'ehs_ranking' not in df.columns:
            raise ValueError("DataFrame must have 'ehs_ranking' column. Run score_dataframe() first.")

        ranking_counts = df['ehs_ranking'].value_counts()
        total = len(df)

        return {
            'total_chemicals': total,
            'recommended': ranking_counts.get('recommended', 0),
            'problematic': ranking_counts.get('problematic', 0),
            'hazardous': ranking_counts.get('hazardous', 0),
            'recommended_pct': ranking_counts.get('recommended', 0) / total * 100,
            'problematic_pct': ranking_counts.get('problematic', 0) / total * 100,
            'hazardous_pct': ranking_counts.get('hazardous', 0) / total * 100,
            'avg_health_score': df['health_score'].mean(),
            'avg_env_score': df['env_score'].mean(),
            'avg_safety_score': df['safety_score'].mean()
        }

    def get_detailed_report(self, df: pd.DataFrame, smiles_col: str = 'smiles') -> pd.DataFrame:
        """
        Get a detailed report with key columns for analysis.

        Args:
            df: DataFrame with all scores
            smiles_col: Name of the SMILES column

        Returns:
            DataFrame with selected columns for reporting
        """
        report_cols = [
            smiles_col, 'IUPAC', 'trivial_name',
            'health_score', 'corrected_health_score',
            'env_score', 'corrected_env_score',
            'safety_score', 'fp_score', 'ait_score', 'bp_score',
            'ehs_ranking',
            'black_health_count', 'red_health_count', 'orange_health_count',
            'yellow_health_count', 'green_health_count',
            'black_env_count', 'red_env_count', 'yellow_env_count'
        ]

        # Add physical properties if available
        physical_props = ['ExpBP', 'FP', 'AIT', 'cLogP', 'mw']
        for prop in physical_props:
            if prop in df.columns:
                report_cols.append(prop)

        # Filter to existing columns
        report_cols = [col for col in report_cols if col in df.columns]

        return df[report_cols].sort_values('ehs_ranking',
                                           ascending=False)  # Hazardous first
