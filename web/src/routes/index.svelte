<script lang="ts">
  import { greet } from '$lib/asplib'

  let greeting = greet()
</script>

<h1 class="bg-blue-300">Hello SvelteKit</h1>
<h1 class="bg-red-300">{greeting}</h1>
