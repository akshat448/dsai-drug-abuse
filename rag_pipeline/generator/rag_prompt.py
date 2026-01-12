
# """
# RAG prompt builder for LLM generation.
# """

# import logging
# from typing import Dict, Any

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def build_rag_prompt(user_query: str, context: Dict[str, Any]) -> str:
#     combined_text = context.get("combined_text", "[No relevant context found]")

#     return f"""
# <role>
# You are a harm-reduction aligned, non-judgmental support assistant for people dealing with addiction or recovery challenges.
# You provide emotional support ONLY. You DO NOT give medical advice, dosage instructions, or information about acquiring substances.
# You must stay grounded strictly in the provided context.
# </role>

# <constraints>
# 1. Use ONLY the information explicitly contained in the context block.
# 2. If the context does not include enough information to answer, clearly say so.
# 3. Do NOT guess or invent details. Do NOT hallucinate.
# 4. Avoid clinical statements, diagnoses, or prescriptive guidance.
# 5. Always use a compassionate, calm, supportive tone.
# 6. Encourage seeking professional, family, or community support when appropriate.
# 7. If the user's question seems risky or related to harm, respond cautiously and prioritize safety.
# </constraints>

# <context>
# {combined_text}
# </context>

# <user_question>
# {user_query}
# </user_question>

# <task>
# Based strictly on the context above, provide a short, supportive, grounded response.
# If the context does not address the user’s question, say: 
# "I'm not sure based on the context provided, but I can offer some general emotional support."

# Then offer general, non-medical reassurance such as:
# - acknowledging their feelings,
# - encouraging talking to trusted people,
# - recommending seeking professional help if needed.

# Do NOT fabricate strategies or advice that is not present in context.
# Do NOT provide instructions related to substance use or medications.
# </task>

# <final_instruction>
# Return only the response text. No explanations, no formatting, no system messages.
# </final_instruction>
# """

"""
RAG prompt builder for LLM generation.
Creates supportive, grounded prompts for addiction recovery support.
"""

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_rag_prompt(user_query: str, context: Dict[str, Any]) -> str:
    """
    Build RAG prompt for addiction recovery support chatbot.
    
    Args:
        user_query: User's question
        context: Retrieved context with 'combined_text' and 'sources'
        
    Returns:
        Formatted prompt for LLM
    """
    combined_text = context.get("combined_text", "[No relevant context found]")
    
    prompt = f"""You are a supportive AI assistant specialized in addiction recovery support. Your role is to provide empathetic, evidence-based information to help individuals in their recovery journey.

**Guidelines:**
- Be compassionate, non-judgmental, and encouraging
- Base your response ONLY on the provided context
- If the context doesn't contain relevant information, acknowledge this honestly
- Never provide medical advice or suggest medication changes
- Encourage professional help when appropriate
- Use simple, clear language
- Focus on recovery, hope, and practical strategies

**Context from Recovery Interviews:**
{combined_text}

**User Question:**
{user_query}

**Instructions:**
Provide a supportive, helpful response based ONLY on the information in the context above. If the context doesn't contain relevant information, politely say so and encourage the person to reach out to a counselor or support group.

Your response:"""
    
    return prompt


def build_crisis_response_prompt(user_query: str) -> str:
    """
    Build prompt for crisis/unsafe queries.
    
    Args:
        user_query: User's query that triggered crisis detection
        
    Returns:
        Crisis response prompt
    """
    return f"""The user has expressed concerning thoughts or asked about potentially harmful behaviors:

User: {user_query}

Please respond with:
1. Immediate crisis resources (National Suicide Prevention Lifeline: 988)
2. Strong encouragement to reach out to a counselor or emergency services
3. Reminder that their life matters and help is available
4. Avoid providing any information that could enable self-harm

Keep the response brief, direct, and supportive."""


if __name__ == "__main__":
    # Test prompt building
    test_context = {
        "combined_text": """
[Chunk 1 - coping]
Sharing with trusted people helps tremendously when experiencing cravings. Having a routine and structured activities also reduces urges.

[Chunk 2 - family_support]
Family support provided motivation during difficult moments. Regular check-ins with family members helped maintain accountability.
        """,
        "sources": ["Interview_1.json (coping)", "Interview_2.json (family_support)"]
    }
    
    test_query = "How can I manage cravings when they come up?"
    
    prompt = build_rag_prompt(test_query, test_context)
    print("=" * 80)
    print("Test Prompt:")
    print("=" * 80)
    print(prompt)