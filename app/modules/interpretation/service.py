import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import re
import dateparser
from dateutil.relativedelta import relativedelta

from app.core.config import settings
from app.modules.interpretation.schemas import InterpretationContext, InterpretationResult, InterpretationData

logger = logging.getLogger(__name__)

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

class InterpretationService:
    def __init__(self):
        logger.info(f"Loading QA Model: {settings.QA_MODEL_NAME}...")
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=settings.QA_MODEL_NAME, 
            tokenizer=settings.QA_MODEL_NAME
        )
        
        logger.info(f"Loading Embedding Model: {settings.EMBEDDING_MODEL_NAME}...")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info("Models loaded successfully.")

    def _get_current_context_string(self) -> str:
        now = datetime.now()
        return f"Today is {now.strftime('%Y-%m-%d')}. The user's prompt is:"

    def _ask_qa(self, question: str, context: str) -> Tuple[Optional[str], int, int]:
        """Runs the QA pipeline and returns (answer, start_index, end_index)"""
        try:
            result = self.qa_pipeline(question=question, context=context)
            answer = result['answer'].strip()
            

            return answer, result['start'], result['end']
        except Exception as e:
            logger.error(f"QA Error for question '{question}': {e}")
            return None, 0, 0

    def _match_entity(self, text: str, candidates: Dict[str, str]) -> Tuple[Optional[str], Optional[str], float]:
        """
        Matches text against candidate dict {id: name}. 
        Returns (id, name, score).
        If score < threshold, returns (None, text, score).
        """
        if not text or not candidates:
            return None, text, 0.0

        candidate_ids = list(candidates.keys())
        candidate_names = list(candidates.values())

        # 1. Exact/Substring Match (Case-insensitive)
        text_lower = text.lower()
        for idx, name in enumerate(candidate_names):
            name_lower = name.lower()
            # Check for exact match or strong substring match
            if name_lower == text_lower or name_lower in text_lower or text_lower in name_lower:
                return candidate_ids[idx], name, 1.0
        
        # 2. Semantic Search
        query_emb = self.embedder.encode(text, convert_to_tensor=True)
        corpus_emb = self.embedder.encode(candidate_names, convert_to_tensor=True)
        
        hits = util.semantic_search(query_emb, corpus_emb, top_k=1)
        
        if hits and hits[0]:
            best_hit = hits[0][0]
            score = best_hit['score']
            if score > settings.MATCH_THRESHOLD:
                idx = best_hit['corpus_id']
                return candidate_ids[idx], candidate_names[idx], score
            else:
                return None, text, score
        
        return None, text, 0.0

    def _normalize_frequency(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        text = text.lower()
        
        # Check for explicit recurrence keywords
        is_recurring = any(x in text for x in ["every", "each", "recurring", "repeatedly"])
        
        if "monthly" in text or (is_recurring and "month" in text):
            return "MONTHLY"
            
        # For weekly, check for "weekly" OR (recurrence keyword + any day/week reference)
        if "weekly" in text or (is_recurring and any(x in text for x in ["week"] + DAYS)):
            return "WEEKLY"

        if "daily" in text or (is_recurring and "day" in text):
            return "DAILY"
            
        return "ONCE" # Default fallback

    def _normalize_currency(self, text: Optional[str]) -> str:
        text_lower = text.lower().strip()
        
        if any(x in text_lower for x in ["eur", "euro", "€"]):
            return "EUR"
            
        return "NOT_SUPPORTED"
    
    def _validate_day(self, text: Optional[str]) -> Optional[str]:
        if not text: return None
        for d in DAYS:
            if d in text.lower():
                return d.capitalize()
        return None

    def _calculate_end_date(self, start_date_str: Optional[str], duration_text: Optional[str]) -> Optional[str]:
        """Calculates end date based on start date and duration string (e.g. 'for a month')."""
        if not start_date_str: 
            return None
            
        if duration_text and ("goal" in duration_text.lower() or "until" in duration_text.lower()):
            return None
        
        try:
            start_dt = dateparser.parse(start_date_str)
            if not start_dt:
                return None
                
            if duration_text and "month" in duration_text:
                end_dt = start_dt + relativedelta(months=1)
                return end_dt.strftime("%Y-%m-%d")
            
            return None
        except:
            return None

    def interpret(self, context: InterpretationContext) -> InterpretationResult:
        time_context = self._get_current_context_string() 
        group_names = ", ".join(context.user_groups.values()) if context.user_groups else "None"
        goal_names = ", ".join(context.user_goals.values()) if context.user_goals else "None"
        
        augmented_text = (
            f"Analyze this request for saving details.\n"
            f"User's Request: {context.prompt}\n"
            f"Time context: {time_context}\n"
        ).lower()

        final_group_id = None
        final_group_name = None
        is_existing_group = False
        
        final_goal_id = None
        final_goal_name = None
        is_existing_goal = False
        
        intent = "personal_saving" # Default
        
        found_group_direct_match = False
        if context.user_groups:
            prompt_lower = context.prompt.lower()
            for g_id, g_name in context.user_groups.items():
                if g_name.lower() in prompt_lower:
                    intent = "group_saving"
                    final_group_id = g_id
                    final_group_name = g_name
                    is_existing_group = True
                    found_group_direct_match = True
                    break
        
        if not found_group_direct_match:
            intent_text, i_start, i_end = self._ask_qa("Does the user want to save for a group or for themselves? Answer only with 'group_saving' or 'personal_saving'.", augmented_text)
            
            if intent_text == "group_saving":
                intent = "group_saving"
                target_text, t_start, t_end = self._ask_qa("What is the specific name of the    group being saved for, from the User's Request?", augmented_text)
            
            else:
                intent = "personal_saving"
                target_text, t_start, t_end = self._ask_qa("What is the specific name of the item or objective being saved for, from the User's Request?", augmented_text)

            if target_text:
                # Validation: Check if day of week
                if target_text and any(d in target_text.lower() for d in DAYS):
                     target_text = None

            if target_text:
                # Check against Groups first
                g_id, g_name, g_score = self._match_entity(target_text, context.user_groups)
                # Check against Goals
                gl_id, gl_name, gl_score = self._match_entity(target_text, context.user_goals)
                
                # Prioritize Goal match if strong
                if gl_id and gl_score > g_score:
                    intent = "personal_saving"
                    final_goal_id = gl_id
                    final_goal_name = gl_name
                    is_existing_goal = True
                elif g_id:
                     intent = "group_saving"
                     final_group_id = g_id
                     final_group_name = g_name
                     is_existing_group = True
                else:
                    # New Entity. Check context for "group" keyword fallback
                    if "group" in context.prompt.lower():
                         intent = "group_saving"
                         final_group_name = "NON_EXISTENT"
                    else:
                         intent = "personal_saving"
                         # Capitalize first letter only
                         final_goal_name = target_text[0].upper() + target_text[1:] if target_text else target_text
            else:
                # If QA matched nothing, but "group" is in prompt
                if "group" in context.prompt.lower():
                     intent = "group_saving"
                     final_group_name = "NON_EXISTENT"

        logger.info(f"Determined Intent: {intent}")

        amount_text, _, _ = self._ask_qa("What is the specific monetary value or numerical amount to be saved from the User's Request?", augmented_text)
        amount_val = None
        if amount_text:
            match = re.search(r"[\d,]+(\.\d+)?", amount_text)
            if match:
                try:
                    amount_val = float(match.group(0).replace(',', ''))
                except:
                    pass

        currency_text, _, _ = self._ask_qa("What is the currency associated with the amount from the User's Request?", augmented_text)
        currency = self._normalize_currency(currency_text)

        frequency_text, _, _ = self._ask_qa("How often should the deposit be made(e.g. daily, weekly, monthly), based on the User's Request?", augmented_text)
        frequency = self._normalize_frequency(frequency_text)
        
        day_text, _, _ = self._ask_qa("Which specific day of the week (e.g. Monday, Tuesday) is mentioned in the User's Request?", augmented_text)
        day_of_week = self._validate_day(day_text)
        
        start_date_text, _, _ = self._ask_qa("What is the specific start date or starting time mentioned in the User's Request?", augmented_text)
        
        duration_text, _, _ = self._ask_qa("What is the duration, end date, or stopping condition for the saving in the User's Request?", augmented_text)
        end_date = self._calculate_end_date(start_date_text, duration_text)

        data = InterpretationData(
            amount=amount_val,
            currency=currency,
            frequency=frequency,
            day_of_week=day_of_week,
            start_date=start_date_text,
            end_date=end_date,
            is_existing_group=is_existing_group,
            group_name=final_group_name,
            group_id=final_group_id,
            is_existing_goal=is_existing_goal,
            goal_name=final_goal_name,
            goal_id=final_goal_id
        )

        return InterpretationResult(
            intent=intent,
            data=data,
            raw_prompt=context.prompt
        )
