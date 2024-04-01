import aiohttp
import asyncio
from jsonargparse import CLI
from jsonargparse.typing import final
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import re
from tqdm.asyncio import tqdm as tqdm_asyncio
import logging

logging.basicConfig(level=logging.INFO)


@final
@dataclass
class Label(object):
    name: str
    related: list[str] = field(default_factory=list)


class ConceptNet(object):
    def __init__(self, http: aiohttp.ClientSession):
        self.cache = {}
        self.URL = "https://api.conceptnet.io/query"
        self.http = http

    async def _fetch(self, params: dict) -> dict:
        async with self.http.get(self.URL, params=params) as response:
            return await response.json()

    async def query(
        self, cls_name: str, with_cache: bool = True, forbidden: list[str] = []
    ) -> set[str]:
        if with_cache and cls_name in self.cache:
            return self.cache[cls_name]

        all_concepts = set()
        params = {
            "node": f"/c/en/{cls_name}",
        }

        rels = [
            "/r/HasA",
            "/r/MadeOf",
            "/r/HasProperty",
            "/r/IsA",
            "/r/PartOf",
            "/r/CapableOf",
            "/r/RelatedTo",
        ]
        objs = await asyncio.gather(
            *(self._fetch(params={**params, "rel": rel}) for rel in rels)
        )
        iter_edges = lambda obj: (edge for edge in obj["edges"])
        filter_en = lambda edges: (
            edge
            for edge in edges
            if edge["start"].get("language", "en") == "en"
            and edge["end"].get("language", "en") == "en"
        )
        iter_labels = lambda edges, pos: (edge[pos]["label"] for edge in edges)

        for obj in objs:
            all_concepts.update(iter_labels(filter_en(iter_edges(obj)), "end"))
            all_concepts.update(iter_labels(filter_en(iter_edges(obj)), "start"))

        all_concepts = (c.lower() for c in all_concepts)
        # Drop the "a "  and "an " prefix for concepts defined like "a {concept}".
        all_concepts = (re.sub(r"\b(?:a |an )\b", "", c) for c in all_concepts)
        # Drop all empty concepts.
        all_concepts = (c for c in all_concepts if c != "")
        # Replace all spaces with underscores.
        all_concepts = (c.replace(" ", "_").replace("-", "_") for c in all_concepts)
        # Drop all concepts that are the same as the class name.
        if cls_name not in forbidden:
            forbidden.append(cls_name)
        all_concepts = (c for c in all_concepts if c not in forbidden)
        # Make each concept unique in the set.
        self.cache[cls_name] = all_concepts = set(all_concepts)
        return all_concepts


async def get_concepts(
    label: Label, concept_net: ConceptNet, with_cache: bool
) -> set[str]:
    forbidden = [label.name, *label.related]
    tasks = [
        concept_net.query(label.name, with_cache=with_cache, forbidden=forbidden),
        *map(concept_net.query, label.related),
    ]
    res = await asyncio.gather(*tasks)
    return list(set().union(*res))


async def get_concept_data(
    labels: list[Label], with_cache: bool
) -> dict[str, set[str]]:
    label_names = [label.name for label in labels]
    async with aiohttp.ClientSession() as http:
        concept_net = ConceptNet(http)
        tasks = (
            get_concepts(label, concept_net, with_cache=with_cache) for label in labels
        )
        concept_data =  dict(zip(label_names, await tqdm_asyncio.gather(*tasks)))
        empty = False
        # if any concept is empty, print it and raise exception
        for label, concepts in concept_data.items():
            if not concepts:
                logging.error(f"No concepts found for {label}")
                empty = True
        if empty:
            raise ValueError("No concepts found for some labels") 
        return concept_data



def main(labels: list[Label], output: Path, with_cache: bool = True):
    res = asyncio.run(get_concept_data(labels, with_cache=with_cache))
    with open(output, "w") as f:
        yaml.dump(res, f)


if __name__ == "__main__":
    CLI(main)
