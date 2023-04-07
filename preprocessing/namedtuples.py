from collections import namedtuple

Article = namedtuple(
    typename='Article',
    field_names=['article_id', 'title', 'language', 'sentences', 'techniques'],
    defaults=([]),  # `defaults` apply to the two rightmost fields
)

Line = namedtuple(
    typename='Line',
    field_names=['article_id', 'line_number', 'text', 'language', 'techniques'],
)

Span = namedtuple(
    typename='Span',
    field_names=['article_id', 'start', 'end', 'technique'], # note technique is singular, only one possible
    defaults=(''),  # `defaults` apply to the two rightmost fields
)

Result = namedtuple(
    typename='Result',
    field_names=['article_id', 'paragraph_id', 'techniques', 'language'], # language is necessary to then split submission labels into correct languages
)