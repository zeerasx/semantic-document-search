def hit_rate(
    retrieved_texts,
    expected_keyword
):

    return int(

        any(

            expected_keyword.lower()
            in text.lower()

            for text in retrieved_texts
        )
    )


def precision_at_k(
    retrieved_texts,
    expected_keyword,
    k=3
):

    retrieved = retrieved_texts[:k]

    hits = sum(

        expected_keyword.lower()
        in text.lower()

        for text in retrieved
    )

    return hits / k


def recall(
    retrieved_texts,
    expected_keyword
):

    return int(

        any(

            expected_keyword.lower()
            in text.lower()

            for text in retrieved_texts
        )
    )


def reciprocal_rank(
    retrieved_texts,
    expected_keyword
):

    for rank, text in enumerate(
        retrieved_texts,
        start=1
    ):

        if (
            expected_keyword.lower()
            in text.lower()
        ):

            return 1 / rank

    return 0