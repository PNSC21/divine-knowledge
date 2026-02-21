function goToVerse(select, chapterId) {
    const verseId = select.value;
    if (verseId) {
        window.location.href = `/chapter/${chapterId}/verse/${verseId}`;
    }
}
