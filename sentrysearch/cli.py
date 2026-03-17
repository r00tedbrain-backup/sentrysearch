"""Click-based CLI entry point."""

import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
def cli():
    """Search dashcam footage using natural language queries."""


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--chunk-duration", default=30, help="Chunk duration in seconds.")
@click.option("--overlap", default=5, help="Overlap between chunks in seconds.")
def index(path, chunk_duration, overlap):
    """Index video files from PATH for searching."""
    import os
    from .chunker import chunk_video, scan_directory
    from .embedder import embed_chunks
    from .store import get_collection, store_embeddings

    if os.path.isdir(path):
        videos = scan_directory(path)
    else:
        videos = [path]

    click.echo(f"Found {len(videos)} video(s) to index.")
    collection = get_collection()

    for video_path in videos:
        click.echo(f"Processing {video_path}...")
        chunks = chunk_video(video_path, chunk_duration=chunk_duration, overlap=overlap)
        click.echo(f"  Created {len(chunks)} chunk(s).")
        embeddings = embed_chunks(chunks)
        store_embeddings(collection, embeddings)
        click.echo(f"  Indexed {len(embeddings)} chunk(s).")

    click.echo("Indexing complete.")


@cli.command()
@click.argument("query")
@click.option("-n", "--num-results", default=5, help="Number of results to return.")
def search(query, num_results):
    """Search indexed footage with a natural language QUERY."""
    from .search import search_clips

    results = search_clips(query, n_results=num_results)
    if not results:
        click.echo("No results found.")
        return

    for i, result in enumerate(results, 1):
        click.echo(f"\n--- Result {i} ---")
        click.echo(f"  Source: {result['source_file']}")
        click.echo(f"  Time:  {result['start_time']:.1f}s - {result['end_time']:.1f}s")
        click.echo(f"  Score: {result['score']:.4f}")


@cli.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--start", required=True, type=float, help="Start time in seconds.")
@click.option("--end", required=True, type=float, help="End time in seconds.")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output file path.")
def trim(video, start, end, output):
    """Trim a clip from VIDEO between --start and --end seconds."""
    from .trimmer import trim_clip

    trim_clip(video, start, end, output)
    click.echo(f"Saved clip to {output}")
