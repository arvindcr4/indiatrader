
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class NLPFeatureGenerator:

    """Generate NLP features for text data."""

    def generate_features(self, df: pd.DataFrame, feature_config: List[Dict]) -> pd.DataFrame:
        result_df = df.copy()
        for spec in feature_config:
            name = spec.get("name")
            params = spec.get("params", {})
            if name == "sentiment_score":
                result_df = self._add_sentiment_score(result_df)
            elif name == "news_embedding":
                result_df = self._add_embedding(result_df, params.get("dimensions", 16))
            else:
                logger.warning(f"Unknown NLP feature: {name}")
        return result_df

    def _add_sentiment_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            logger.warning("Text column not found for sentiment analysis")
            return df
        positive = {"good", "great", "positive", "up", "bull", "gain"}
        negative = {"bad", "down", "negative", "bear", "loss"}

        def score(text: str) -> float:
            tokens = text.lower().split()
            pos = sum(1 for t in tokens if t in positive)
            neg = sum(1 for t in tokens if t in negative)
            total = len(tokens) if tokens else 1
            return (pos - neg) / total

        df["sentiment_score"] = df["text"].fillna("").apply(score)
        return df

    def _add_embedding(self, df: pd.DataFrame, dims: int) -> pd.DataFrame:
        if "text" not in df.columns:
            logger.warning("Text column not found for embeddings")
            return df

        def embed(text: str):
            digest = hashlib.sha256(text.encode()).digest()
            return [int(b) for b in digest[:dims]]

        df["news_embedding"] = df["text"].fillna("").apply(embed)
        return df
=======
    """
    Generate NLP features from text data.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize NLP feature generator.
        
        Args:
            cache_dir: Directory to cache embeddings and sentiment scores
        """
        self.models = {}
        self.tokenizers = {}
        
        # Set cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache", "nlp")
        else:
            self.cache_dir = cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def generate_features(self, 
                         text_df: pd.DataFrame, 
                         feature_config: Dict,
                         text_column: str = "text",
                         timestamp_column: str = "timestamp") -> pd.DataFrame:
        """
        Generate NLP features based on configuration.
        
        Args:
            text_df: DataFrame with text data
            feature_config: Feature configuration dictionary
            text_column: Name of column containing text
            timestamp_column: Name of column containing timestamps
        
        Returns:
            DataFrame with added NLP features
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = text_df.copy()
        
        # Validate required columns
        if text_column not in result_df.columns:
            logger.error(f"Text column '{text_column}' not found in DataFrame")
            return result_df
        
        if timestamp_column not in result_df.columns:
            logger.warning(f"Timestamp column '{timestamp_column}' not found. Using DataFrame index.")
        
        # Generate features for each configured feature type
        for feature_spec in feature_config:
            feature_type = feature_spec["name"]
            model_name = feature_spec.get("model")
            params = feature_spec.get("params", {})
            
            if feature_type == "sentiment_score":
                result_df = self._add_sentiment_score(result_df, text_column, model_name, **params)
            
            elif feature_type == "news_embedding":
                result_df = self._add_text_embeddings(result_df, text_column, model_name, **params)
            
            elif feature_type == "keyword_extraction":
                result_df = self._add_keyword_features(result_df, text_column, **params)
            
            elif feature_type == "topic_modeling":
                result_df = self._add_topic_features(result_df, text_column, **params)
            
            elif feature_type == "entity_recognition":
                result_df = self._add_entity_features(result_df, text_column, **params)
            
            else:
                logger.warning(f"Unknown feature type: {feature_type}")
        
        return result_df
    
    def _add_sentiment_score(self, 
                            df: pd.DataFrame, 
                            text_column: str,
                            model_name: str = "FinBERT",
                            max_length: int = 512,
                            batch_size: int = 16) -> pd.DataFrame:
        """
        Add sentiment scores to text data.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            model_name: Name of sentiment model to use
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing
        
        Returns:
            DataFrame with added sentiment features
        """
        # Check if cached results exist
        cache_file = os.path.join(self.cache_dir, f"sentiment_{model_name.lower().replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.json")
        
        # Try to load from cache
        sentiment_cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    sentiment_cache = json.load(f)
                logger.info(f"Loaded {len(sentiment_cache)} cached sentiment scores")
            except Exception as e:
                logger.warning(f"Failed to load sentiment cache: {str(e)}")
        
        # Analyze texts not in cache
        texts_to_analyze = []
        text_indices = []
        
        for i, row in df.iterrows():
            text = row[text_column]
            
            # Skip empty texts
            if pd.isna(text) or text.strip() == "":
                continue
            
            # Use cached result if available
            if text in sentiment_cache:
                continue
            
            texts_to_analyze.append(text)
            text_indices.append(i)
        
        # Process new texts
        if texts_to_analyze:
            logger.info(f"Analyzing sentiment for {len(texts_to_analyze)} new texts")
            
            if model_name.lower() == "finbert" and TRANSFORMERS_AVAILABLE:
                # Load FinBERT model if not already loaded
                if "finbert" not in self.models:
                    try:
                        self.tokenizers["finbert"] = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                        self.models["finbert"] = AutoModel.from_pretrained("ProsusAI/finbert")
                        logger.info("Loaded FinBERT model")
                    except Exception as e:
                        logger.error(f"Failed to load FinBERT model: {str(e)}")
                        # Use a simple rule-based approach as fallback
                        return self._add_rule_based_sentiment(df, text_column)
                
                # Process texts in batches
                model = self.models["finbert"]
                tokenizer = self.tokenizers["finbert"]
                
                all_scores = []
                
                for i in range(0, len(texts_to_analyze), batch_size):
                    batch_texts = texts_to_analyze[i:i+batch_size]
                    
                    # Tokenize and prepare inputs
                    inputs = tokenizer(batch_texts, 
                                      return_tensors="pt", 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=max_length)
                    
                    # Get model outputs
                    with torch.no_grad():
                        outputs = model(**inputs)
                        
                    # For FinBERT, we need to apply the classification head
                    # This is a simplified version - the actual implementation would be more complex
                    last_hidden_states = outputs.last_hidden_state
                    cls_embeddings = last_hidden_states[:, 0, :]
                    
                    # Apply a simple sentiment classification
                    # This is a placeholder - normally you'd use the actual classification head
                    sentiment_direction = torch.sum(cls_embeddings, dim=1)
                    
                    # Convert to sentiment scores between -1 and 1
                    scores = torch.tanh(sentiment_direction / 10).cpu().numpy()
                    all_scores.extend(scores.tolist())
                
                # Add new scores to cache
                for text, score in zip(texts_to_analyze, all_scores):
                    sentiment_cache[text] = score
            
            elif model_name.lower() == "text-embedding-3-large" and OPENAI_AVAILABLE:
                # Use OpenAI API for sentiment analysis
                try:
                    all_scores = []
                    
                    for text in texts_to_analyze:
                        prompt = f"Analyze the sentiment of this financial news text. Output only a single number between -1 (extremely negative) and 1 (extremely positive):\n\n{text}"
                        
                        response = openai.Completion.create(
                            model="text-embedding-3-large",
                            prompt=prompt,
                            max_tokens=1,
                            temperature=0
                        )
                        
                        try:
                            score = float(response.choices[0].text.strip())
                            score = max(-1, min(1, score))  # Ensure score is between -1 and 1
                        except ValueError:
                            score = 0  # Default to neutral if parsing fails
                        
                        all_scores.append(score)
                    
                    # Add new scores to cache
                    for text, score in zip(texts_to_analyze, all_scores):
                        sentiment_cache[text] = score
                
                except Exception as e:
                    logger.error(f"Failed to use OpenAI API: {str(e)}")
                    # Use a simple rule-based approach as fallback
                    return self._add_rule_based_sentiment(df, text_column)
            
            else:
                # Fallback to rule-based approach
                logger.warning(f"Model {model_name} not available or not supported. Using rule-based sentiment analysis.")
                return self._add_rule_based_sentiment(df, text_column)
            
            # Save updated cache
            try:
                with open(cache_file, "w") as f:
                    json.dump(sentiment_cache, f)
                logger.info(f"Saved {len(sentiment_cache)} sentiment scores to cache")
            except Exception as e:
                logger.warning(f"Failed to save sentiment cache: {str(e)}")
        
        # Apply sentiment scores to DataFrame
        df["sentiment_score"] = df[text_column].apply(
            lambda text: sentiment_cache.get(text, 0) if pd.notna(text) and text.strip() != "" else 0
        )
        
        # Add derived sentiment features
        df["sentiment_positive"] = np.where(df["sentiment_score"] > 0.2, 1, 0)
        df["sentiment_negative"] = np.where(df["sentiment_score"] < -0.2, 1, 0)
        df["sentiment_neutral"] = np.where(
            (df["sentiment_score"] >= -0.2) & (df["sentiment_score"] <= 0.2), 1, 0
        )
        
        # Calculate rolling sentiment features
        window_sizes = [5, 10, 20]
        
        for window in window_sizes:
            df[f"sentiment_ma_{window}"] = df["sentiment_score"].rolling(window=window).mean()
            df[f"sentiment_std_{window}"] = df["sentiment_score"].rolling(window=window).std()
            
            # Sentiment momentum
            df[f"sentiment_momentum_{window}"] = df[f"sentiment_ma_{window}"] - df[f"sentiment_ma_{window}"].shift(window)
            
            # Sentiment regime
            df[f"sentiment_regime_{window}"] = np.where(
                df[f"sentiment_ma_{window}"] > 0.1, "positive",
                np.where(df[f"sentiment_ma_{window}"] < -0.1, "negative", "neutral")
            )
        
        return df
    
    def _add_rule_based_sentiment(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Add rule-based sentiment scores as a fallback.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
        
        Returns:
            DataFrame with added sentiment features
        """
        logger.info("Using rule-based sentiment analysis")
        
        # Define positive and negative word lists
        positive_words = [
            "up", "rise", "gain", "profit", "growth", "increase", "positive", "bullish", "rally",
            "outperform", "beat", "strong", "strength", "opportunity", "promising", "optimistic",
            "upside", "advantage", "succeed", "successful", "improved", "improvement", "grew",
            "upgrade", "higher", "support", "backed", "confident", "breakthrough", "triumph"
        ]
        
        negative_words = [
            "down", "fall", "drop", "loss", "decline", "decrease", "negative", "bearish", "plunge",
            "underperform", "miss", "weak", "weakness", "risk", "concerning", "pessimistic",
            "downside", "disadvantage", "fail", "failed", "deteriorated", "deterioration", "shrank",
            "downgrade", "lower", "resistance", "opposed", "worried", "setback", "disappointment"
        ]
        
        # Define negation words
        negation_words = ["not", "no", "never", "neither", "nor", "none", "hardly", "barely", "scarcely"]
        
        # Define intensifier words
        intensifiers = ["very", "extremely", "highly", "significantly", "substantially", "strongly", 
                        "notably", "markedly", "considerable", "greatly", "sharply"]
        
        # Calculate sentiment scores
        def calculate_sentiment(text):
            if pd.isna(text) or text.strip() == "":
                return 0
            
            # Lowercase and split text
            text = text.lower()
            words = re.findall(r'\b\w+\b', text)
            
            score = 0
            skip_next = False
            
            for i, word in enumerate(words):
                if skip_next:
                    skip_next = False
                    continue
                
                # Check for intensifiers
                intensity = 1.0
                if i > 0 and words[i-1] in intensifiers:
                    intensity = 1.5
                
                # Check for negation
                negated = False
                for neg_idx in range(max(0, i-3), i):
                    if words[neg_idx] in negation_words:
                        negated = True
                        break
                
                # Calculate word sentiment
                if word in positive_words:
                    score += intensity * (-1 if negated else 1)
                elif word in negative_words:
                    score -= intensity * (-1 if negated else 1)
            
            # Normalize score to range [-1, 1]
            if score != 0:
                score = score / (abs(score) + 5)  # Dampening factor
            
            return score
        
        # Apply sentiment calculation
        df["sentiment_score"] = df[text_column].apply(calculate_sentiment)
        
        # Add derived sentiment features
        df["sentiment_positive"] = np.where(df["sentiment_score"] > 0.2, 1, 0)
        df["sentiment_negative"] = np.where(df["sentiment_score"] < -0.2, 1, 0)
        df["sentiment_neutral"] = np.where(
            (df["sentiment_score"] >= -0.2) & (df["sentiment_score"] <= 0.2), 1, 0
        )
        
        # Calculate rolling sentiment features
        window_sizes = [5, 10, 20]
        
        for window in window_sizes:
            df[f"sentiment_ma_{window}"] = df["sentiment_score"].rolling(window=window).mean()
            df[f"sentiment_std_{window}"] = df["sentiment_score"].rolling(window=window).std()
            
            # Sentiment momentum
            df[f"sentiment_momentum_{window}"] = df[f"sentiment_ma_{window}"] - df[f"sentiment_ma_{window}"].shift(window)
            
            # Sentiment regime
            df[f"sentiment_regime_{window}"] = np.where(
                df[f"sentiment_ma_{window}"] > 0.1, "positive",
                np.where(df[f"sentiment_ma_{window}"] < -0.1, "negative", "neutral")
            )
        
        return df
    
    def _add_text_embeddings(self, 
                           df: pd.DataFrame, 
                           text_column: str,
                           model_name: str = "text-embedding-3-small",
                           dimensions: int = 1536,
                           batch_size: int = 32,
                           use_pca: bool = True,
                           pca_components: int = 20) -> pd.DataFrame:
        """
        Add text embeddings to text data.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            model_name: Name of embedding model to use
            dimensions: Dimensions of the embedding
            batch_size: Batch size for processing
            use_pca: Whether to reduce embedding dimensions with PCA
            pca_components: Number of PCA components if use_pca is True
        
        Returns:
            DataFrame with added embedding features
        """
        # Check if cached results exist
        cache_file = os.path.join(self.cache_dir, f"embeddings_{model_name.lower().replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.npz")
        
        # Try to load from cache
        embedding_cache = {}
        if os.path.exists(cache_file):
            try:
                npz_data = np.load(cache_file, allow_pickle=True)
                texts = npz_data["texts"]
                embeddings = npz_data["embeddings"]
                
                for text, embedding in zip(texts, embeddings):
                    embedding_cache[text] = embedding
                
                logger.info(f"Loaded {len(embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {str(e)}")
        
        # Process texts not in cache
        texts_to_embed = []
        text_indices = []
        
        for i, row in df.iterrows():
            text = row[text_column]
            
            # Skip empty texts
            if pd.isna(text) or text.strip() == "":
                continue
            
            # Use cached result if available
            if text in embedding_cache:
                continue
            
            texts_to_embed.append(text)
            text_indices.append(i)
        
        # Process new texts
        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} new texts")
            
            if "text-embedding" in model_name.lower() and OPENAI_AVAILABLE:
                # Use OpenAI API for embeddings
                try:
                    all_embeddings = []
                    
                    for i in range(0, len(texts_to_embed), batch_size):
                        batch_texts = texts_to_embed[i:i+batch_size]
                        
                        response = openai.Embedding.create(
                            model=model_name,
                            input=batch_texts
                        )
                        
                        batch_embeddings = [data["embedding"] for data in response["data"]]
                        all_embeddings.extend(batch_embeddings)
                    
                    # Add new embeddings to cache
                    for text, embedding in zip(texts_to_embed, all_embeddings):
                        embedding_cache[text] = embedding
                
                except Exception as e:
                    logger.error(f"Failed to use OpenAI API: {str(e)}")
                    # Use a simple fallback approach
                    return self._add_tfidf_embeddings(df, text_column)
            
            elif TRANSFORMERS_AVAILABLE:
                # Use Hugging Face transformers for embeddings
                try:
                    # Use a suitable model based on the requested one
                    if "small" in model_name.lower():
                        hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    else:
                        hf_model_name = "sentence-transformers/all-mpnet-base-v2"
                    
                    # Load model if not already loaded
                    if hf_model_name not in self.models:
                        self.tokenizers[hf_model_name] = AutoTokenizer.from_pretrained(hf_model_name)
                        self.models[hf_model_name] = AutoModel.from_pretrained(hf_model_name)
                        logger.info(f"Loaded {hf_model_name} model")
                    
                    model = self.models[hf_model_name]
                    tokenizer = self.tokenizers[hf_model_name]
                    
                    all_embeddings = []
                    
                    for i in range(0, len(texts_to_embed), batch_size):
                        batch_texts = texts_to_embed[i:i+batch_size]
                        
                        # Tokenize and prepare inputs
                        inputs = tokenizer(batch_texts, 
                                          return_tensors="pt", 
                                          padding=True, 
                                          truncation=True, 
                                          max_length=512)
                        
                        # Get model outputs
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        # Use mean pooling to get sentence embeddings
                        attention_mask = inputs["attention_mask"]
                        token_embeddings = outputs.last_hidden_state
                        
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        
                        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                        all_embeddings.extend(batch_embeddings.tolist())
                    
                    # Add new embeddings to cache
                    for text, embedding in zip(texts_to_embed, all_embeddings):
                        embedding_cache[text] = embedding
                
                except Exception as e:
                    logger.error(f"Failed to use transformers: {str(e)}")
                    # Use a simple fallback approach
                    return self._add_tfidf_embeddings(df, text_column)
            
            else:
                # Fallback to TF-IDF approach
                logger.warning(f"Model {model_name} not available or not supported. Using TF-IDF embeddings.")
                return self._add_tfidf_embeddings(df, text_column)
            
            # Save updated cache
            try:
                all_texts = list(embedding_cache.keys())
                all_embeddings = list(embedding_cache.values())
                
                np.savez_compressed(
                    cache_file,
                    texts=np.array(all_texts, dtype=object),
                    embeddings=np.array(all_embeddings, dtype=object)
                )
                logger.info(f"Saved {len(embedding_cache)} embeddings to cache")
            except Exception as e:
                logger.warning(f"Failed to save embedding cache: {str(e)}")
        
        # Create embedding matrix
        all_embeddings = []
        
        for i, row in df.iterrows():
            text = row[text_column]
            
            if pd.isna(text) or text.strip() == "":
                # Use zero vector for empty texts
                embedding = np.zeros(dimensions if "text-embedding" in model_name.lower() else 
                                     (768 if "mpnet" in self.models else 384))
            else:
                embedding = embedding_cache.get(text, np.zeros(dimensions))
            
            all_embeddings.append(embedding)
        
        # Convert to numpy array
        embedding_matrix = np.vstack(all_embeddings)
        
        # Apply PCA if requested
        if use_pca and len(df) > pca_components:
            try:
                from sklearn.decomposition import PCA
                
                pca = PCA(n_components=pca_components)
                reduced_embeddings = pca.fit_transform(embedding_matrix)
                
                # Add PCA components to DataFrame
                for i in range(pca_components):
                    df[f"embedding_pca_{i+1}"] = reduced_embeddings[:, i]
                
                # Calculate embedding norms and other features
                df["embedding_norm"] = np.linalg.norm(reduced_embeddings, axis=1)
                
                # Calculate embedding similarity to market regime prototypes
                # This is a placeholder - in a real system, you'd have actual prototypes
                bullish_prototype = np.ones(pca_components) / np.sqrt(pca_components)
                bearish_prototype = -bullish_prototype
                
                df["embedding_bullish_sim"] = np.dot(reduced_embeddings, bullish_prototype)
                df["embedding_bearish_sim"] = np.dot(reduced_embeddings, bearish_prototype)
            
            except Exception as e:
                logger.error(f"Failed to apply PCA: {str(e)}")
                # Add raw embedding components
                max_cols = min(20, embedding_matrix.shape[1])
                for i in range(max_cols):
                    df[f"embedding_{i+1}"] = embedding_matrix[:, i]
        
        else:
            # Add raw embedding components (limited to avoid too many columns)
            max_cols = min(20, embedding_matrix.shape[1])
            for i in range(max_cols):
                df[f"embedding_{i+1}"] = embedding_matrix[:, i]
            
            # Calculate embedding norm
            df["embedding_norm"] = np.linalg.norm(embedding_matrix, axis=1)
        
        return df
    
    def _add_tfidf_embeddings(self, df: pd.DataFrame, text_column: str, max_features: int = 100) -> pd.DataFrame:
        """
        Add TF-IDF based embeddings as a fallback.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            max_features: Maximum number of TF-IDF features
        
        Returns:
            DataFrame with added TF-IDF embedding features
        """
        logger.info("Using TF-IDF embeddings as fallback")
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            # Fill NaN values
            texts = df[text_column].fillna("").tolist()
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
                ngram_range=(1, 2)
            )
            
            # Generate TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Apply SVD for dimensionality reduction
            n_components = min(20, tfidf_matrix.shape[1], len(texts) - 1)
            svd = TruncatedSVD(n_components=n_components)
            reduced_embeddings = svd.fit_transform(tfidf_matrix)
            
            # Add SVD components to DataFrame
            for i in range(n_components):
                df[f"tfidf_svd_{i+1}"] = reduced_embeddings[:, i]
            
            # Calculate embedding norm
            df["tfidf_norm"] = np.linalg.norm(reduced_embeddings, axis=1)
            
            # Add feature for vocabulary richness
            df["vocab_richness"] = np.array([
                len(set(re.findall(r'\b\w+\b', text.lower()))) / (len(re.findall(r'\b\w+\b', text.lower())) + 1)
                for text in texts
            ])
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to generate TF-IDF embeddings: {str(e)}")
            return df
    
    def _add_keyword_features(self, df: pd.DataFrame, text_column: str, 
                            financial_term_lists: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Add keyword-based features.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            financial_term_lists: Dictionary of term categories and their keywords
        
        Returns:
            DataFrame with added keyword features
        """
        # Default financial term lists if not provided
        if financial_term_lists is None:
            financial_term_lists = {
                "bullish": [
                    "rally", "upgrade", "buy", "bullish", "outperform", "overweight", "growth",
                    "upside", "strong", "higher", "positive", "profit", "gain", "opportunity",
                    "recovery", "beat", "exceeded", "expanding", "momentum", "surging"
                ],
                "bearish": [
                    "sell", "bearish", "downgrade", "underperform", "underweight", "slowdown",
                    "downside", "weak", "lower", "negative", "loss", "risk", "concern", "recession",
                    "miss", "missed", "contracting", "slump", "declining", "tumbling"
                ],
                "volatility": [
                    "volatile", "uncertainty", "turbulence", "fluctuation", "instability",
                    "swings", "erratic", "jumpiness", "oscillation", "unpredictable"
                ],
                "merger_acquisition": [
                    "merger", "acquisition", "takeover", "buyout", "acquiring", "acquired",
                    "purchase", "bid", "offer", "deal", "transaction", "consolidation"
                ],
                "regulation": [
                    "regulation", "regulatory", "compliance", "law", "legal", "policy",
                    "regulator", "oversight", "reform", "rule", "requirement", "legislation"
                ],
                "dividend": [
                    "dividend", "yield", "payout", "distribution", "income", "shareholder return"
                ]
            }
        
        # Process each text and count term occurrences
        for category, terms in financial_term_lists.items():
            col_name = f"keyword_{category}"
            
            # Count occurrences of terms in each text
            df[col_name] = df[text_column].apply(
                lambda text: sum(1 for term in terms if term.lower() in text.lower()) if pd.notna(text) else 0
            )
            
            # Normalize by text length
            df[f"{col_name}_norm"] = df.apply(
                lambda row: row[col_name] / (len(row[text_column].split()) + 1) if pd.notna(row[text_column]) else 0,
                axis=1
            )
        
        # Calculate aggregate features
        df["keyword_sentiment"] = df["keyword_bullish"] - df["keyword_bearish"]
        df["keyword_event_flag"] = np.where(
            df["keyword_merger_acquisition"] + df["keyword_regulation"] > 0, 1, 0
        )
        
        # Calculate rolling metrics
        for window in [5, 10, 20]:
            df[f"keyword_sentiment_ma_{window}"] = df["keyword_sentiment"].rolling(window=window).mean()
            df[f"keyword_volatility_ma_{window}"] = df["keyword_volatility"].rolling(window=window).mean()
        
        return df
    
    def _add_topic_features(self, df: pd.DataFrame, text_column: str, num_topics: int = 5) -> pd.DataFrame:
        """
        Add topic modeling features.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            num_topics: Number of topics to extract
        
        Returns:
            DataFrame with added topic features
        """
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            
            # Fill NaN values
            texts = df[text_column].fillna("").tolist()
            
            # Create and fit CountVectorizer
            vectorizer = CountVectorizer(
                max_features=1000,
                stop_words="english"
            )
            
            X = vectorizer.fit_transform(texts)
            
            # Create and fit LDA model
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10
            )
            
            # Get document-topic distributions
            doc_topic_dists = lda.fit_transform(X)
            
            # Add topic distributions to DataFrame
            for i in range(num_topics):
                df[f"topic_{i+1}"] = doc_topic_dists[:, i]
            
            # Add dominant topic
            df["dominant_topic"] = np.argmax(doc_topic_dists, axis=1) + 1
            
            # Add topic diversity (entropy-based)
            def calculate_entropy(probs):
                return -np.sum(probs * np.log2(probs + 1e-10))
            
            df["topic_diversity"] = np.apply_along_axis(calculate_entropy, 1, doc_topic_dists)
            
            # Add topic shift features (changes in topics over time)
            df["topic_shift"] = np.zeros(len(df))
            
            for i in range(1, len(df)):
                prev_dist = doc_topic_dists[i-1]
                curr_dist = doc_topic_dists[i]
                
                # Calculate Jensen-Shannon divergence
                m = 0.5 * (prev_dist + curr_dist)
                js_div = 0.5 * (calculate_entropy(m) - 0.5 * (calculate_entropy(prev_dist) + calculate_entropy(curr_dist)))
                
                df.loc[df.index[i], "topic_shift"] = js_div
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to generate topic features: {str(e)}")
            return df
    
    def _add_entity_features(self, df: pd.DataFrame, text_column: str, 
                           company_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add named entity recognition features.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            company_list: List of company names to look for
        
        Returns:
            DataFrame with added entity features
        """
        # Basic implementation using regex patterns for companies, numbers, and dates
        
        # Company pattern (simplified)
        company_pattern = r"\b[A-Z][a-zA-Z\&]+(?:\s+(?:Inc|Corp|Ltd|Limited|LLC|Group|Holdings|Technologies|Pharmaceuticals|Bank|Systems|Solutions|International))?\.?\b"
        
        # Percentage pattern
        percentage_pattern = r"\b\d+(?:\.\d+)?%\b"
        
        # Money pattern
        money_pattern = r"\$\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|mn|bn|tn))?\b"
        
        # Date pattern
        date_pattern = r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b"
        
        # Count entity occurrences
        df["entity_companies"] = df[text_column].apply(
            lambda text: len(set(re.findall(company_pattern, text))) if pd.notna(text) else 0
        )
        
        df["entity_percentages"] = df[text_column].apply(
            lambda text: len(re.findall(percentage_pattern, text)) if pd.notna(text) else 0
        )
        
        df["entity_money"] = df[text_column].apply(
            lambda text: len(re.findall(money_pattern, text)) if pd.notna(text) else 0
        )
        
        df["entity_dates"] = df[text_column].apply(
            lambda text: len(re.findall(date_pattern, text)) if pd.notna(text) else 0
        )
        
        # Check for specific companies if provided
        if company_list is not None:
            df["entity_target_companies"] = df[text_column].apply(
                lambda text: sum(1 for company in company_list if company.lower() in text.lower()) if pd.notna(text) else 0
            )
        
        # Calculate total entity density
        df["entity_density"] = df.apply(
            lambda row: (row["entity_companies"] + row["entity_percentages"] + 
                         row["entity_money"] + row["entity_dates"]) / 
                         (len(row[text_column].split()) + 1) if pd.notna(row[text_column]) else 0,
            axis=1
        )
        
        # Identify financial report patterns
        earnings_pattern = r"\b(?:earnings|revenue|profit|income|EPS|EBITDA|guidance|forecast|outlook)\b"
        df["is_earnings_related"] = df[text_column].apply(
            lambda text: 1 if pd.notna(text) and len(re.findall(earnings_pattern, text, re.IGNORECASE)) > 0 else 0
        )
        
        return df

